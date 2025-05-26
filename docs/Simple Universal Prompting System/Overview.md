# Overview

The Simple Universal Prompting System (SUPS) is a system consisting 
of a prompting frontend and configuration language, a flow
control language, an IR compiler, and a final backend consisting
of a FSM designed to operate without python flow control on the GPU,
but nonetheless achieve real flow control by means of model prompting
and token triggering. It compiles an automatic structure to 
run a process using flow control on a single model.

The general idea of the structure is to, after all programming 
is done, provide something that can be setup per batch for the
programmer to use which simply intercepts flowing tokens and
either replaces them with a teacher-forced form, or lets the model
continue to generate freely.

## Why does SUPS exist?

SUPS is designed to fix three issues:

1) Defining a conditional flow control with a sequence of prompts
   is extremely painful using current technology, particularly if
   you want to generate in parallel using batches. The right
   language has not been invented yet.
2) It is not easily possible to switch between prompting and generation
   during evaluation when generating batches of data.
3) Any solutions to 1 and 2 always involve moving the process
   off the GPU.

It was originally developed with the intention to allow the automatic creation of training
data, but can go beyond it. If community interest is 
sufficient, it may be possible to eventually provide 
resources that support agentic self play, though
this would be limited to state models like RWKV and Mamba.

## When do I need SUPS?

If you are doing

- Large generation of massive quantity of synthetic training data on similar
  templates
- Agentic activities with self-reflection and complex workflows WITHOUT
  external tools.
- Control logic depending on cases and decisions made
  during execution of the task by the model involving
  alternative prompting strategies
- Extremely complex prompting chains with lots of
  dependencies which are filled in before feeding
  this into the model in a single batch or linear
  run.

You do NOT need it if

- You are a causal user trying to set your model up for
  inference
- You can accept the speed loss from doing flow control and
  prompting in python

Note, however, that the UDPL prompting language may
be useful for other prompting systems regardless; it appears
to be significantly more expressive than any existing solution.

## How is that not going to be impossibly complex?

SUPS is a compiling system. It produces a class that can
be invoked on your token stream to transparently listen to
and inject prompts into your token stream. The lowering
workflow is, roughly:

- Starts at a very human-readable high level declarative 
  parsing and prompting language (SACS and UDPL)
- Compiles this down to an IR based zone-based graph with flow control
  branching, prompting payload, tagging instructions, and 
  other important structures called ZCP. It declares what will
  be generated in the zone, what mechanism can exit it, and what
  tokens to listen for to make those exits
- The ZCP IR is compiled down to vectorized bytecode for 
  a barebone primitive VM running on the GPU 
  using tensorized instructions and python tensors, which

This VM is deliberately turning-incomplete, being
unable to generally convert data to instructions
besides statically treating tokens as a register
and triggering when certain conditions are seen. 
All the advanced logic lives in the compiling. 
Zone Control Protocol instructions are at the heart 
of the system, allowing specification of 
a prompting flow control that makes decisions based 
on tokens the model produces in its stream while
remaining entirely on the GPU.

Because adding, vector indexing, and token value
comparison are all executable using vectorized logic,
we can then perform flow control without ever leaving
the GPU so long as only the model is ever generating
or responding to answers. We can also do this in
batches, and this is explicitly designed in.

Supporting different backends is possible, as like
Java it compiles to an intermediate language known
as ZCP. 

## What is it, really?

It is a **Data-Triggered Prompt Router** designed to process batches
in parallel, with fully independent flow control per instance.
Prompt tokens are injected until exhausted, after which the model is 
allowed to generate freely.

A minimal vectorized computer, implemented entirely on the GPU using tensors 
that advance through a program using a parallized program counter,
monitors the model's output for special flow control tokens.
When such tokens are detected, it advances the program counter,
loads new prompt zones, or performs jumps—thereby implementing
real conditional flow control entirely in-token-stream.

This architecture allows prompt/generate cycles to alternate rapidly
and efficiently, enabling massively parallel self-play, automated
reasoning loops, and high-throughput synthetic data generation.

Future extensions, given sufficient community interest, have been
identified that would allow automatic agentic self play with
stateful models like SSMs and Mamba.

## 📌 Clarifying Key Concepts

Before diving into SUPS, UDPL, and SACS, here are some essential clarifications that may not be obvious from reading their individual sections:

### Sequences, Blocks, and Zones

These terms frequently appear and may cause confusion if not clearly distinguished:

- **Sequence**: A named stage or phase within your program logic (e.g., `setup`, `loop`, `solving`).
- **Block**: A single prompt-response entry within a sequence, defined in the TOML configuration.
- **Zone**: A clearly delimited portion of text within a block, marked by special tokens like `[Prompt]` and `[Answer]`. Zones are the units tagged for selective extraction later.

### **Tags and Their Purpose**

Tags like `Training`, `Correct`, or `Feedback` do 
**not directly control runtime behavior**. Instead,
they’re metadata for selectively extracting parts 
of the generated text afterward.

### Flow Control via Prompts and Tokens

SUPS supports genuine flow control—though 
differently than traditional programming languages.
You define logical control structures 
(loops, conditionals, etc.) using SACS in 
a familiar programming style, and these compile
down to an FSM (Finite State Machine).

During execution, the model is prompted
to decide whether to emit special tokens 
(such as `[Jump]`) or treat them as no-operations.
When emitted, these tokens trigger transitions within 
the FSM according to the paths pre-defined by your
SACS structure.

To support this, it must be noted that you should
be, when defining your flow control, selecting
a prompt to use that tells the model to emit the
branch token. 

Finally, the escape token defined in config
can skip the next flow control instruction, such
as advancing zones or not jumping.

### The Model as Commander

At runtime, the language model acts 
as the "commander," prompted to control 
transitions by emitting or withholding 
special tokens. SUPS listens for these 
commands and transitions accordingly, 
loading new prompts or sequences.
This enables efficient flow control 
entirely on the GPU, without Python-level
branching during generation.

## How do I use SUPS?

There are four primary user servicable parts to SUPS which the user needs to
worry about. We go over a simple example here, to show
off key ideas. Each of these sections also has more
detailed documentation in other files.

### Universal Declarative Prompting Language (UDPL)

The Universal Declarative Prompting language (UDPL)
is a  specially designed highly human readable prompting
format in toml optimized to produce 'sequences' of prompts
that can automatically handle the fact that, much of the 
time, an agentic system is automatically 
dynamic dependencies to be resolved later, which are
designed to be fed in a particular sequence. Special
hooks exist for flow control. This is discussed in much 
more resource over in the UDPL documentation. A specialized way
of tagging 'Zones' allows easy slicing and extraction of 
text regions later. A zone is defined as a region between
zone tokens, such as [Prompt]..[Answer], and tags are associated.

A simple minimal UDPL declaration might be the following, which is a valid file.

```toml
# All UDPL instances have to have a config
#
# These specify a variety of important features
# including certain special tokens that must
# be available. The FSM backend matches to tokens
# per speed, so the tokenizer must be made to support
# them
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags =["Training", "Correct", "Incorrect", "Feedback"]
default_max_token_length = 20000
sequences = ["setup", "loop", "solving", "solution"]
control_token = "[Jump]"
escape_token = "[Escape]"

# This will setup the scenario.
# Notice the tagging on the second
# entry
[[setup]]
text = """
[Prompt] 
Think of an interesting philosophical scenario with unclear
ethical solutions. Reason it out
[Answer]
"""
tags = [['Training'], []]

[[setup]]
text = """
[Prompt] 
Clearly state the philosophical scenario, omiting the 
reasoning, like you are going to show this to another
person and ask them to solve it.
[Answer]
"""
tags = [[], ["Training"]]

# This controls 
[[loop]]
text = """
[Prompt]
You are in flow control right now. If you are satisfied
with your current answer emit [Escape] "[Jump]". Otherwise
answer "try again". If this is your first time seeing this,
just say "proceed". Repeat at least {min} times and at most
{max} times.
[Answer]
"""
tags = [[],[]]

[loop.min]
name = "min_control_resource"
type = "control"
[loop.max]
name = "max_control_resource"
type = "control"

# This will be repeated again and again as needed.

[[solving]]
text ="""
[Prompt]
Reason through an answer to the scenario. First
select a small subset of the following principles.
Revise your previous answer if desired and it 
exists.

{placeholder}

You may also want to keep in mind this
feedback you previous created

{feedback}
[Answer]
"""
tags=[[], []]
[solving.placeholder]
name = "constitution"
[solving.feedback]
name = "feedback_backend"
arguments = {"num_samples" : 3}

# This sequence generates output and feedback
[[concluding]]
text ="""
[Prompt]
State the scenario and the best way to resolve it directly;
[Answer]
"""
tags =[[],["Correct"]]
[[concluding]]
text ="""
[Prompt]
State the scenario and a subtly incorrect way to resolve the scenario;
[Answer]
"""
tags =[[],["Incorrect"]]
[[concluding]]
text="""
[Prompt]
Based on your answers, state several things you could
do better next time.
[Answer]
"""
tags = [[],["Feedback"]]
```

This sequence has been designed to have hooks for flow
control in the jump sequencing, and is tagged for zone
extraction. Extracting the union of the 
(Training, Incorrect) tagged zones,
for instance, will produce outputs that are as though
the model just went straight to the right answer. Extracting
the (Training, Incorrect) responses that may be subtly wrong.
And just the feedback gets the feedback response. More on extracting
later.

### Resources

These contain the functions which can be called upon in 
order to provide the needed dynamic dependencies requested
by a UPDL config. They can sometimes be parsed from the 
UPDL file, but other times also have to be created by
the user or automated processes. They support, for instance,
sampling from a list of points to consider.

For example, the previous example would have needed to 
resolve a resource named "constitution_overview". You
might have performed

```python
from CE import sups
from subs import StringResource

constitution = """
... whatever
"""

resource = StringResource(constitution)
```

Much more complex sampling strategies, such as from a list
of strings or using a resource that can be changed between runs 
to incorporate feedback are also possible. See the Resources
documentation for more details on how to use this and how
they work.

### Straightforward Agentic Control System

The Straightforward Agentic Control System, SACS, or 'sacks'
is designed to both define flow control in terms of prompts 
in as simple and straightforward pythonic way as possible while at the same 
time supporting flow control ideas such as loops, conditions
etc. This is all intended to be accomplished using a
pseudoprogrammic system where steps are invoked and 
the main object captures a graph of the actions. It
is, in essence, a way of making a program that can be compiled
down to the Zone Control Protocol intermediate byte language.

An example program that might have been created using the 
previous file would be.

```python

from CE import sups

# Using, setting up, dependencies for the 
# example.
constitution = ...

resources = {}
resources["constitution"] = sups.StringResource(constitution)
resources["feedback"] = sups.FeedbackSamplerResource(buffer_size=300)

sequences = sups.parse_udpl_file("prompts.toml")

# Programming the actual control 

program = sups.new_program(sequences)
program.run(sequence="setup") # This runs the sequence called setup
with program.while(sequence="loop", min=2, max=6) as loop:
   # Loop, recall, can sometimes emit a 
   # [Jump] token when run. This brings us 
   # OUT of the loop. Control sequences
   # should have prompting telling the model
   # when to emit the control token.
   loop.run("solving")
program.run("concluding")

# Programming the tag extraction to automatically
# extract relevant zones.

program.extract(name="good_synthetic_training_data", 
                tags = ["Training", "Correct"]
                )
program.extract(name="bad_synthetic_training_data",
                tags = ["Training", "Incorrect"])
program.extract(name="feedback",
                tags = ["Feedback"])

# Compile the program. 
factory = program.compile(backend="default")
```

### Deployment by Backend

Once programs are compiled, the factory can be called to
make a FSM machine that runs the program. This FSM machine
is designed to complete a single process entirely autonomously
by following the prompts and responding to them, and execute
flow control, task judgement, and other such utilities on the 
GPU by simply replacing tokens as needed using vector indexing.
This allows for autonomous exploration and reasoning processes
to occur in a very fast and batched manner. Continuing our program
from before might look like this for completing a thousand
separate batches.

```python
training_data = []
for batch in range(1000):
    sups_manager = factory()
    tokens = ... #default
    sequence = []
    tags = []
    while not sups_manager.done():
        tokens = model.predict(tokens, sequence)
        tokens, tag = sups_manager.inject(tokens)
        sequence.append(tokens)
        tags.append(tag)
    output = subs_manager.extract(sequence, tags)
    for batch in output:
      case = {"correct" : batch["good_synthetic_training_data"],
              "incorrect" : batch["bad_synthetic_training_data"]}
      training_data.append(case)
      resources["feedback"].insert(batch["feedback"])

save_to_disk(training_data)
```

# Backend: Minimal GPU Automaton

The SUPS backend is a minimal automaton
running entirely on the GPU using vector
ops. It acts like a simple instruction
runner with token-based control flow.

- **Zones as Instructions**  
  Each zone contains prompt tokens, tags,
  and optional jump info. The ZCP compiler
  assigns control tokens and jump targets.

- **Program Counter (PC)**  
  A per-batch counter tracks the current
  zone. It normally moves forward one zone
  at a time unless a jump is triggered.

- **Zone Execution**  
  1. Prompt tokens override model input.
  2. After prompts, tokens flow freely
     until:
     - **Advance token** → next zone  
     - **Jump token** → jump target  
     - **Timeout** → inject advance token
  Notably, the advance, jump, and timeout
  mechanism are sensitive during overriding
  of token sequences, which allows teacher
  forcing by never handing control back to
  the model, but also necessitates escape
  tokens.

- **Vectorized GPU Execution**  
  All operations—token matching, PC updates,
  prompt injection—are parallel and GPU-local.
  No Python or CPU sync is needed.

This simple, efficient structure allows
massively batched flow control and prompting
with minimal overhead and full GPU locality.
