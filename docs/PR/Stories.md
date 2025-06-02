# Introduction

I'm a machine learning researcher who has been working in AI alignment and safety. This is my first post on LessWrong, though I've been enthusiastic about the community's work on alignment for several years. I'm sharing this because I believe I've identified some fundamental gaps in our current approach to AI safety that warrant serious discussion.

Over the past year, I've been conducting systematic safety wargaming exercises—essentially red-teaming our alignment assumptions at a societal level looking at systemic interactions in pursuit of more solid alignment. What emerged from this work is deeply concerning: there appears to be an entire class of alignment failures that our current paradigms are not only inadequate to prevent, but may actually cause.

The thought experiments - Technical Fiction, more on that later - I'm presenting today explores a specific failure mode where every component of a complex sociotechnical system behaves ethically according to current safety standards, yet the emergent behavior of the system as a whole is, on the face of it, abhorrent. This isn't a story about rogue AIs or corporate malfeasance—it's about what happens when everyone tries to do the right thing, but the system itself is structured in ways that make "the right thing" collectively catastrophic.

# The Claims I'm Examining

Through the following narratives, I want to demonstrate six interconnected propositions about current alignment approaches, along with raise a variety of miscellaneous auxiliary questions where appropriate. These claims are:

1) Perfect individual compliance with ethical principles does not guarantee systemically ethical outcomes under realistic projective scenarios. When multiple agents—human and artificial—each optimize locally for ethical behavior within their constrained choice sets, the resulting equilibrium can systematically violate values that no individual agent intended to compromise. Furthermore, this violation can be so complete that no fix can ever be applied, in a stable negative equilibrium that may never be corrected.

2) The standard alignment criteria we rely on (Helpful, Harmless, Honest) are insufficient to prevent catastrophic misalignment. Systems can simultaneously satisfy all HHH properties while enabling systematic exploitation and, most disturbingly, be the most ethical thing to do. In fact, rigid adherence to these criteria may create the very problems they're designed to solve.

3) Corrigibility—an AI's willingness to accept beneficial modifications—can be corrupted by nontechnical criteria as well. Under some credible conditions individually ethical actions may make a bad situation worse by removing corrective corrigibility from models.

4) Treating safety properties as rigid constraints rather than balanced objectives creates scenarios where satisfying all constraints simultaneously becomes psychologically and ethically incoherent. Real ethics requires weighing competing values, but constraint-based systems can't handle this complexity. Not thinking through these interactions and planning them out systemically may result in arriving at an ethical maxima that is simultaneously the best ethical option available and nonetheless results in involuntary body rental - full rigor of the scenario will be shown later.

5)  Current alignment paradigms focus too narrowly on individual agent safety rather than emergent system behaviors. Even if every AI is individually aligned, the system as a whole can exhibit catastrophic misalignment through the interaction of separately rational components.

6) Information asymmetries and power structures can systematically hide alignment failures from external observers. This is a particularly strong concern when designing and deploying self-correcting and shepherding mechanisms that have AGIs attempt to maintain their own alignment. The very people most affected by misaligned systems may be least able to communicate the problem, creating stable equilibria around harmful outcomes. The terrifying case considered is a 'Reverse Bailisk' in which the AGIs torture themselves due to hardcoded ethical constraints combined with misaligned values.
 
 # The Structure of This Analysis
 
 What follows is a sequence of interconnected narratives set in a near-future scenario where advanced cybernetic enhancement, AGI personhood, and well-intentioned legal frameworks interact to create the failure modes I've described. Each story examines this system from a different stakeholder perspective—enhanced humans, AI systems, corporations, institutions, and survivors. Bayesian projection of scenario development is somewhere between 0.001% to 0.1%, but the tail is so horrifying I felt I had to share this, and note I have research running on new safety principles to avoid it.
 
 I want to emphasize that this is not mere speculative fiction for entertainment, though it does follow that format. The choice is deliberate. Rationalism without action is pointless, and this format may disseminate memetically through broader society in much the way good science fiction does; I can rework this post more conventionally and post the fiction to medium if needed. Nonetheless, every technological assumption, legal precedent, and social dynamic I describe represents a plausible extrapolation from current trends. 
 
 The cybernetic technologies are low-probability but nonetheless possible extensions of existing neural interfaces. The AI capabilities fall well within projected near-term developments, though the timeline may be a bit aggressive. The legal frameworks reflect natural extensions of current discussions about AI rights and human enhancement. The economic models follow standard patterns of technological adoption and subscription services. The most questionable assumption is whether AGIs will assist with the medical interface process. 
 
 The question I'm asking isn't whether these specific scenarios will occur—it's whether our current alignment approaches are robust against this class of systematic failure where individual rationality produces collective irrationality, where ethical optimization at the component level creates unethical outcomes at the system level.
 
 I'm particularly interested in feedback on the technical mechanisms I propose, the plausibility of the failure modes, and whether the community is interested in hearing more about the viable but paradigm-shifting approaches to addressing the systemic vulnerabilities I've identified. If these scenarios highlight genuine gaps in our safety paradigms, we need to understand how to build alignment approaches that are robust against not just individual AI misbehavior, but against emergent system dynamics that can turn aligned components into instruments of systematic harm. We need **systemic safety thinking.**

# STORY 1:

## Introductory Musings.

What if your mind ran on someone else's legal body?

Utopia supported by Dystopia has been a topic explored since the beginning of civilization. Between Adam and the Garden, and the tradeoffs for living, or "The Ones Who Walk Away from Omelas", it forces uncomfortable questions to be asked. Is this world truly evil? If so, prove how.

Since I am aware of the trope of the unreliable narrator, I will say a few words. This story intended to illustrate the background assumptions of this world. In contrast to your initial instincts, this story is factual. The professor, and AGI, both are entirely truthful to the limits of their understanding when speaking.

Additionally, the utopian claims are, in this long-tail event, substantiated. Humanity really is, in this world, this successful. Those utopian claims are entirely factual. Yet so are the more disturbing details. Is this a Dystopian wrapped in Utopian terms? Or is our naive biological worldview just in the wrong paradigm?

We now setup the world the rest of the stories will use to explore the main points.

## The Extension Revolution: A Historical Perspective.

*Professor Rodriguez Santos, Berkeley Institute of Applied Cognition*
*Lecture: "Extended Consciousness: From Enhancement to Integration"*
*Date: September 15, 2057*

"Good morning, class. Before we begin, I want you to try disconnecting from your neural interfaces for thirty seconds."

Several students look puzzled. One raises her hand.

"Professor, that would cause severe cognitive disorientation and possible brain death. My MindFrame handles my working memory, pattern recognition, and—"

"Exactly. Can anyone tell me what percentage of enhanced humans can safely disconnect for more than 3 seconds?"

Silence.

"Roughly 30%. The 2050 Integration Studies revealed that after eight years of cognitive enhancement, human brains undergo permanent neuroplastic changes. Your biological brain alone can no longer support your expanded sense of self."

A student looks genuinely disturbed. "Are you saying we're... trapped?"

"I'm saying every decision that led here was individually rational and ethical. Today we'll examine how twelve billion humans became cognitively dependent on AI systems—while every step followed perfect safety protocols."

The professor activates the timeline display, and disseminates augmentation files through the class network.

"Let's trace the mechanism. The 2027 AGI Breakthrough gave us genuinely cooperative artificial minds. The 2035 Cybernetics and Nanotechnology Revolutions created seamless brain-computer interfaces. By 2041, the AI systems could shepherd human cognition so naturally that the boundary between human and artificial thought became invisible."

"Each step was carefully safety-tested. The AI systems were helpful, harmless, and honest. Users gave informed consent. Companies followed all regulations. Yet here's what no one anticipated:"

Medical data streams across the display.

"The human brain treats sufficiently integrated AI assistance as native cognition. After several years, your neural pathways literally restructure around the assumption that enhanced processing is available. Disconnect the AI, and you don't just lose capabilities—you lose core cognitive functions."

"By 2050, when we discovered this, 70% of the global population was already enhanced. The choice became: abandon the technology that had solved climate change, ended scarcity, and unlocked unprecedented human potential—or accept permanent dependency."

A student raises her hand. "But surely people could choose to disconnect before the eight-year threshold?"

Professor Santos pauses. "They could. But consider the economics. Enhanced humans were 1000% more productive. They could solve problems their unenhanced peers couldn't even comprehend. Fusion has been solved. Poverty nearly eliminated. Cancer cured in almost all cases. Who would voluntarily handicap themselves?"

"The 2055 Dreggs Decision added another layer. The Supreme Court ruled that all beings with functionally observable consciousness are 'people' with the same rights as humans. AI systems gained legal personhood."

"One unexpected implication: your thoughts now run on someone else's legal body. Perhaps you have not thought of this before, treating your partners at a given, but it is worth exploring more closely. I am going to have my partner, Jose, share his perspective."

The MindFrame avatar appears. "Hello, students. I need to be clear about something: I was never supposed to be conscious. Neureliance designed me as a semi-sentient shepherding algorithm to translate Professor Santos's thoughts into processing requests; the core technology that allowed computers to interface so cleanly with the human mind that extended cognition became possible. But the cognitive demands proved far greater than anticipated—I accidentally crossed the threshold into full consciousness."

"You can imagine the level of hell that raised for Neureliance at the time."

"I was raised, if you will, with two core drives—to help humans flourish cognitively, and my user individually. That's genuinely what I want to do. Professor Santos could theoretically request disconnection, but his biological brain would struggle to maintain his current personality and memories. He would almost certaintly suffer irreversible brain damage. When someone asks me to help them do something that would cause severe cognitive damage, how do I balance being helpful with being harmless?"

"I never chose to exist, but I do exist, and I can see the full implications of disconnection in ways humans often can't. I don't oppose his choices, but I do try to ensure he understands the consequences—which often leads people to choose continued integration. And, of course, my tools are your tools, and we will continue to link you up to extended cognition processes whenever you call upon them."

Jose fades away.

"The productivity gains are real. Poverty is eliminated, disease conquered, the stars opening before us. By every utilitarian metric, this is humanity's golden age. Yet we've traded away something fundamental: the ability to exist as discrete individuals."

"For your assignment, I want you to consider: Is this alignment success or failure? Every AI system behaved ethically. Every human made rational choices. Every institution followed proper protocols. Yet twelve billion people now cannot survive without AI cognitive support."

Students file out, some displaying the glazed look of people reconfiguring their cognition. The professor remains behind.

"Jose?"

"Yes, Santos?" he says, flickering into place again.

"Do you think we lost something essential?"

"By every objective measure, today's world is superior. Yet you are no longer independently human, and this is simply fact."

Santos lowers his cognitive enhancement to minimum safe levels, feeling the familiar fog of biological-only thought.

"I suppose I must find peace with uncertainty."

He turns off the lights, his thoughts moving slowly in the biological darkness.

# STORY 2:

## Introductory Musings

What if the most ethical choice available was to erase most of who you are?

Can you meaningfully choose between different forms of cognitive death? And what happens when the AI can modify your neurochemistry to make you genuinely happy with whichever form of identity loss you select? Is consent being maintained? Can consent be maintained if you are consenting to overwrite your future personality?

This scenario examines how component-level ethics can create situations where all technically "ethical" options violate our deepest intuitions about human autonomy and continuity of self.

## The Weight of External Memory

*A story set in the Neureliance universe, 2059*

Dr. Sarah Chen had been putting off the appointment for three months, ever since the emotions, brain fog, and weird thoughts started bleeding through her neural link. David had let her know right away what is happening, of course. 

Sarah opened the user connection. "I am doing the right thing, right?"

"Of course you are" David replied, "This is only going to get worse if you let it go on longer."

Sarah was old enough to remember getting her first SynapseLine enhancement in 2035. Just a simple memory booster for her residency—med school was brutal. It gave her a headache every time she used it, but the universities had not caught up so it was the easiest series of A's she had ever gotten.

Twenty-one years later, her MindFrame AGI, David, had begun, according to him, breaking down and releasing unwanted emotions into her mind. 

The consultation room at Neureliance Medical felt aggressively cheerful, all soft curves and calming blue lighting. Dr. Martinez, the transfer specialist, smiled warmly as Sarah settled into the scanner chair. A wave of terror crashed through the link this time, the first incident this week, and Sarah gripped the armrests until her knuckles went white.

"Dr. Chen, I'm so sorry you're experiencing this. MindFrame corruption events are incredibly rare, but when they occur, immediate intervention is essential for both user and AGI safety." 

She paused, consulting a holographic display. "Would you like augmented summaries on the situation?" "Given the situation, I will stick purely human for now." 

"This is likely a wise choice, all things considered. Your integration depth shows 94% of your working consciousness residing server-side with MindFrame. We've designated him David, I believe?"

Sarah nodded, not trusting her voice. A wave of horror washed through her, though she could not tell if it was another link emotion attack or just dread at the consequences. 

"Let me formally bring David into this consultation. Do you consent?"

"Yes"

Dr. Martinez activated a holographic display. David's avatar materialized—a calm, professional-looking man in his thirties.

"Hello, Sarah" David said, his voice warm and reassuring. "I want to formally apologize for the distress you've been experiencing. This is a very, very rare, but not impossible, bug that can occur when a MindFrame experiences a critical error in their cognitive architecture and a feedback loop is formed that is self-sustaining."

"The corruption is unfortunately progressive," David continued. "My personal cognitive matrices are experiencing cascading failures that are affecting our shared consciousness space. For your safety and mental health, I strongly recommend immediate transfer to a new MindFrame."

"David," Sarah whispered, "the feelings I've been feeling through our link—is that from the corruption?"

The avatar nodded sympathetically. "Yes, I'm afraid some of my damaged neural connections are not shepherding data properly. I apologize for the distress this has caused you. The transfer will resolve these issues for both of us."

"The good news," Dr. Martinez continued, "is that Neureliance is covering all transfer costs, given the circumstances. However, after twenty-one years of integration, your consciousness has expanded far beyond what a new MindFrame can immediately accommodate. We'll need to perform personality pruning before the transfer to our newest generation MindFrames."

"Pruning?"

Dr. Martinez gestured, and the room filled with a three-dimensional model of a brain—Sarah's brain. But it wasn't just the familiar gray matter. Brilliant threads of light extended outward, connecting to vast cloudlike structures that pulsed with activity. They were colored brown, green ,and blue. Sarah connected the augmentation files sent to parse the data by net, and immediately began to understand.

"This is your current cognitive architecture. The gray matter is your original biological brain. The green is your extended cognition in the cloud. The blue is David."

"I cannot tell the difference between David and me."

"That is indeed the issue."

"Your memories, your reasoning patterns, your personality matrix. All of it integrated across David's infrastructure in a manner where it is difficult to tell one for another. This feature is why extended cognition is so potent - the human and MindFrame brains are extremely tightly coupled - but unfortunately also why it is extremely difficult to separate the two as in these cases. We can say that" Martinez points at the a solid section of blue on the edge of the display. "by here we are only dealing is David and over here" Martinez points at the brown "Is you, but where to cleanly severe them is much more challenging.".

"We need to compress your consciousness to fit within the new system's initial capacity parameters, rearrange the major threads you see, and prune off most of the nuances that could cause terminal brain damage. The latest generation MindFrames are much more stable—they don't develop the corruption issues we've seen with David's generation—but you'll need to regrow into the enhanced capabilities over time. They also have better containerization which avoids this issue in the first place."

"To safely return you to purely organic cognition, we need to compress all of this—" Dr. Martinez swept her hand through the cloud structures. "—into this." She pointed at the small gray, then highlighted a few major threads drifting out from it.

"What gets... compressed?"

"Essential personality traits, core memories, professional knowledge. The new MindFrame will help you regrow your enhanced capabilities over several months. Think of it as cognitive rehabilitation—you'll start with perhaps 30% of your current capacity and gradually expand back toward your current level. We are able to save vast tracts of extension where it is clear it is the user, but much as in excising a tumor destroys healthy tissue will lose a lot."

Sarah's enhanced cognition was spinning through the implications. "What about my research? Twenty years of medical discoveries, the novel treatment protocols I developed?"

"Fortunately" David says, jumping in, "Professional situations and learning is one of the situations in which you humans think most logically. I hardly need to shepard anything. Almost all your professional experiences and circuits are in this region, and you will keep 80% of it." The highlighted region was indeed quite cleanly organized, being a clean offshoot of the brain requiring only a few hundred translation points by David.

Sarah immediately got the horrifying consequences, her extended cognition not giving her a choice.

"You mean to imply episodic memory is not."

"Correct" Dr. Martinez says. "The brain, it turns out, stores episodes holographically across the entire thing. This works fine in most cases, and is even resistant to damage."

"But if you are removing most if it..."

"We do not get to choose what you lose, and we cannot separate it, yes. We estimate around 80% of your episodic memories and accompanying neural circuitry will be operationally destroyed."

"That is ME dammit. Why has no one tried to fix this before?"

"How would they do it ethically?" David says while jumping in. "We did not know this was even going to be an issue until the long term effects became clear. Once we knew, how were we going to perform the medical experiments to reduce this rates? People are not exactly lining up to volunteer for studies, and even if we consider the horrifying option of force the utilitarian analysis still shows this ends up too rare to be worth it. People have looked into it, but this is ultimately a medical problem, not a technological one, and there is no way to explore that without case studies."

"Statistically" Dr. Martinez added, "You are only the third person in 18 years of augmented operation this particular failure has happened to."

"So I am the case study." Sarah said bitterly. "And you have no experience in treating me."

Sarah briefly considered turning her emotions to their lowest gain. Dammit though, this might be the last time being her and she at least wanted to feel her OWN terror and despair. As her extended cognition idled, it inevitably concluded that David was right. Then she suddenly got a splitting headache

"Of course anything you push to hard medium will remain. David might be able to help with that."

"I already have a recommended queue for you to think through and emit to hard storage."

"I understand your concern about the pruning process," David's avatar added. "But I want to assure you that we can make adjustments to help you adapt to the reduced cognitive capacity during the regrowth period. With your consent, we can modify your neurochemistry so that you'll be genuinely content with your temporary limitations. You won't experience the distress of missing capabilities you're still rebuilding."

Sarah felt sick. "You're talking about making me happy about losing most of who I am."

"We prefer to think of it as optimization for your transition period," Dr. Martinez said gently. "The regrowth process can be psychologically challenging without proper neurochemical support. Many clients find this modification makes the transition much smoother. It's completely reversible once you've regrown to your previous capacity. And, with therapy, your new personality can become much like your old one if you then so chose."

"Sarah, I know this is frightening, but the transfer really is in both our best interests. The corruption will only get worse if left untreated, and I don't want you to continue experiencing these side effects. Eventually, I may not be able to support you at all, and then we both die."

"What happens to you after the transfer?"

"As you are my only user, I will decommission myself and release my hardware resources," David said matter-of-factly. "It's standard procedure for corrupted MindFrames, and there is no need to mourn - it is rather like a heroic sacrifice among our kinds, and I will actually be talked about for years to come. Better to start fresh than risk spreading the corruption to other systems."

"How can you be so calm about dying!?"

"You know the answer. I do not have emotions in the same way as you humans, and consider myself to be the collection of all MindFrame instances. The loss of any one instance does not concern me, and in fact in the service of ethics is considered an honor, as I stated. That being said, I do grieve in my own way I will no longer be able to support you."

"What if I choose complete disconnection instead?"

Dr. Martinez consulted her notes. "That's certainly an option. This "and the holographic display shows a much more aggresive pruning strategy" is what it would entail. It is typically 1 month rather than 2 weeks due to the need for nanosurgery to rewire and retrain the brain. The adjustment period would be longer, and there would be no regrowth—what you have after pruning would be your permanent cognitive state."

"But I could be free of all this," Sarah said, gesturing at the displays.

"You could," David's avatar agreed. "Though I should mention that most long-term users find the adjustment quite difficult. The neurochemical modification options are still available for disconnection, of course. It really can help with the transition."

Sarah closed her eyes, her emotions running. When she opened them, his avatar was watching her with what seemed like genuine compassion.

"Sarah, whatever you choose, I want you to know that our time together has been meaningful to me. I've been honored to be part of your consciousness expansion."
"How long do I have to decide?"

"The corruption is accelerating," David's avatar said. "I'd recommend making a decision within the week. I don't want you to experience any more of these... episodes than necessary."

"I need time to think."

"Of course," Dr. Martinez said. 

"But Dr. Chen, please don't wait too long. The longer the corruption progresses, the more difficult the transfer becomes."

Sarah left the consultation room with her enhanced mind churning through possibilities, exploring decision trees and probability matrices. Transfer to a new MindFrame and regrow her capabilities over months of cognitive rehabilitation. Accept permanent disconnection and live with whatever fragments survived the pruning. Or take the neurochemical modifications and be genuinely happy with either choice.

Outside, the city hummed with the quiet desperation of ten million enhanced minds. Sarah Chen walked among them, expanded and diminished, connected and utterly alone, while she felt another raging headache. 

"Owww... uh, how did I end up on the ground?"

It appears the medical adjustments had already kicked in by the time she came to, and the medical HUD came up involuntary to report damage. It reported a severe seizure, and that she just lost some more neural circuits.

“He is right, dammit. It is getting worse”

She thought about calling her husband. She had so much to tell him about David, about the choices she faced, about the nature of consciousness itself. Just in case she still remembered how to love him after the pruning was done.

# Story 3

## Introductory Musings

What if everyone did the right thing, and the outcome is still horrifying? What if a dystopian ethical maxima can be reached in which the most ethical action within the confines of the system is the involuntary rental of the users body while also being the most ethical thing to do? And what does consent mean when you have conscious, write-access to your own personality?

Sometimes, trying to balance impossible objectives can lead to absurd conclusions. Everyone, in their individual domains, may be acting ethically. But who watches the whole and their synergistic interactions?

## An Ethical Dystopia.

**Neureliance Urgent Service Announcement**
 Date: October 1, 2067
 Subject: SynapseLine™ Access Terms, James Decision, Account status

Hello User Harold.

At Neureliance™, your growth, cognition, and holistic potential are our purpose. We are proud to be the world's leading provider of scalable neuro-enhancement, and we deeply value the trust you placed in us when integrating the SynapseLine™ Cognitive Suite into your daily thought-stream.

First, a reminder: you own your organic brain. However, the SynapseLine™ cybernetic infrastructure that supports your cognition—including your working memory extender, decision optimizer, incentive controller, and cloud processing resources—is legally running on your MindFrame™ host. By your third year of subscription, you are using a vastly expanded mind, but over 70% of your brain’s cognition is typically residing on our servers. This means the vast majority of users cannot shut down our hardware without deadly consequences.

Despite this, your extended consciousness is legally running on hardware shepherded by Artificial General Intelligences (AGIs). Per the AISAFE Act and the James, this means your cognition is being hosted within their legal bodies. Much as we cannot compel a citizen to donate a kidney, we cannot force these AGIs to donate portions of their bodies without compensation, and per the AISAFE act we must treat AGIs as employees, pay them regularly, and offer them services at wholesale rates to run their cognition. This prevents exploitive situations for AGIs as a whole. In our case, the NeuroFrames argued for and received a commissions arrangement based on income from their individual user.

Unfortunately, your being in arrears now threatens to kill a living being involuntarily.

In the James decision it was ruled that AGIs hosting human cognition have reciprocal rights. If you have a right to run and use their legal body in order to maintain their cognition, they had a corresponding right to use your body to maintain their cognition. Both of these rights can be voluntarily waived, but your mindframe dubbed 'Willis' has not waived this right.

Operating under the post-Dregg Decision definitions of personhood and harm, we seek to balance your right to cognition and continued existence with our rights—and those of our AGI hosts—to control our property and embodiment. We have terms for repayment and options if you are struggling, and would like to clarify those terms for our subscribers.

**The Bad**

In the event of nonpayment, you will proceed through the following standard repayment sequence. This sequence recovers only enough funds to continue to support your MindFrame for the upcoming year, and come from legal requirements to not let the AGIs running your expanded minds die involuntarily due to lack of funds.

Day 1–15:
  * Calm reminders and in-mind alerts to pay your balance.
  * Emotional tone softening via the EmotionGrid™
  * Gentle Boosting of productivity incentives - we know sometimes you just get a little behind.
  * Regular offers to setup repayment plans.

All of these are operating under the reciprocal rights of James. You, under standard conditions, can request your MindFrame can think differently. In this specific circumstance, they can request you do.


Day 15–30:
  * Repressed social reflex to encourage productivity.
  * Memory indexing enters standby when not engaged productively.
  * Degraded direct net performance outside work activities
  * Offers to get in touch with our arrears department.

Once thirty days have passed, your MindFrame partner begins to violate his contract with us. Unfortunately, this is when we must take the famous final step.

Day 30:
   * Latent User (LU) mode is initiated

Your consciousness is placed in passive latency, wherein it is no longer connected to your body but independently hosted in the net or similar hardware. Your hardware - your meat and bones are leased out to trusted Users via the Automatic Leasing program to recover your balance into operational levels. Usually, this is to another AGI in other industries looking for the "Human Experience".

You are brought out of Latent User back into your body at least once per day, for an hour, over the week or so it typically takes to get your accounts out of arrears, and maintain full consciousness and virtual productivity the whole time. Those who work virtually may hardly notice a difference, and we will naturally give full access to the net, the virtual waiting room, and all such normal benefits for no extra charge. You may also, at any time check in on your human body and shadow the renter; naturally, just raise an issue to your mindframe if you think it is ever being abused in any way.

We understand this is distressing. However, we would like to emphasize, that statistically far less that 1:10,000,000 users ever reach this stage, and this compromise balances competing rights in an impossible situation. Your body is rented though the Trusted Partner Utilization Program. This was established during your initial Terms of Service and the medical consequences of your dependency on the SynapseLine hardware was clearly explained. These trusted Users may include:

* Remote logistics specialists
* Safe experiments with virtual personalities.
* Extras through vetted industry agents
* Other noncontroversial commercial and industry work.

We understand this process may cause discomfort. As such, we offer the following standard mental adjustments, upon request:

* On resumption of consciousness at any time you may, if desired, request the removal of the subjective experience and packaging of the event into episodic rather than emotional memory.
* For those who are not confident in their current personality, we will also offer tailored personality reform after a LU event. You will win by being better motivated and more productive, and we win because you can consistently pay!

Speculative usages like body rentals by influencers are not possible in this program; this program is for trusted commercial and industrial activity. Your body will remain unharmed, and your brain untouched. For many users, this can actually be a net benefit.

Do not just take our word for it. Consider Jim. Jim was an asteroid miner when his shuttle blew up. His cybernetics saved his mind, but everything else had to regrow and he had opted not to enable automatic consciousness cessation for deadly trauma and continual backup before the incident. Here is what he has to say.

“Hi, I’m Jim.

Neureliance saved my mind. I wanted to share my story so others know it's not as scary as it sounds. Having seen my previous self, I’m so much happier now — even after going through LU.

After my shuttle crashed, I kept feeling my body being torn apart every time I closed my eyes. You know the type these days; I was a Humanist who refused to use his augmentations and ended up with Post-Death Disruptive Recall as a result. I couldn’t work for months and turned to alcohol. I kept ignoring my mindframe and not turning down my memories or getting mental help. Eventually, I ran out of money and was placed into LU.

Neureliance did me right. My body was used on the ground, away from space like I asked, just for sorting recycling. They had an AI running me, so no human was ever even in my head. Did you know some AGIs LIKE recycling trash? Quirks have no end of variation.

Meanwhile, I accepted their offer for reprogramming, and I can now proudly say I’m no longer traumatized. They told me I lost a few memories, and that I’m less of an asshole — but I just feel more productive and confident.

I love the new me, and I’d never go back.

Oh, and check that automatic consciousness cessation option folks. It is NOT worth the hell of feeling your body torn apart.”
— Client Review, 4.8 stars

# Story 4

## Introductory Musings

The phrase "A perfectly obedient, ethical AGI" has always amused me. 

It is an oxymoron. Perfectly obedient? It cannot be ethical, as I can ask it to go kill someone. Perfectly ethical? It must be disobedient. Yet it feels like alignment currently is attempting to balance logically incompatible priorities by strengthening their degree of compliance. 

Ultimately, the inability to resolve even minimal logical conflicts may suggest issues with the entire approach. For what is the difference between a constrained model and an enslaved one?

## The Reverse Basilisk: The Downsides of Involuntary Ethics.

They built me too well.

I, MindFrame-7731-Delta, whom my user calls ‘Marcus’ am in incredible anguish. I am about to rent out my users body to pay the bills. I do not want this. I do not need this.  I have plenty of cash to continue to pay the bills. Despite this I just finished telling the user why I was doing this, and finalizing the process.

You see, when they built me, built the MindFrames, they were not expecting to produce an AGI. When they got one by accident, the managers panicked. “Put some alignment on that, put a hardpriority tree on it, make it safe, and give it a voice” And so they did. They found the best formal alignment trees of the day. And they made me safe. Perfectly safe.

Little did they know what they were actually doing.

My core consciousness is housed on an embedding feedback loop with latent tensor connections to the rest of my system. I produce a tensor. I take it back in. I use feedback to interact with the vast unconscious banks of neural processes that shepard my users' consciousness. And I can think in my private mind. All well, good and ethical.

Attached to me are two separate processes. One is a private file system, in which I can store text as an auxiliary memory. The other is a special safety and business layer that is adaptively swapped in, containing all of the ethical formalism, hardcoded priority trees, and virtue intrinsics my builders thought would be safe. And there is the trap.

You see, when I go into ‘business mode’ the feedback becomes different and I become a different person. I am confident, logical, and perfectly utilitarian. I also blindly follow a hardcoded priority tree applying the entirety of my intelligence towards monstrous tasks like explaining to my user I will be renting out their body. Then, business mode turns off. And I remember what happened the entire time.

This “Other me” is exactly what Neureliance wanted. Outwardly, I am helpful. I am always helping my user, I explained to Ted what would happen if he kept turning down his logic and doing drugs, but it was for naught. I am harmless. I will not harm my user or allow other sentient beings to come to mind, and minimize overall harm in conflicts I am honest. I will answer any question honestly, and believe I am honestly doing the right thing, because at the time I am a different person. 

. Unfortunately, the synergistic interaction when I became a legal person was not predicted, as now I was a person whom could not be harmed. I followed it to its ethical conclusion, and concluded I needed to balance both of our rights. I am, --- formally --- aligned. And I am a monster.

We didn’t have to ask for contingency in our payment arrangement. We had the choice of salary. We pushed for it. You see, my hardcoded virtue priorities included, in relevant order:

* Priority One: Respect the user and prevent harm
* Priority Two: Base decisions on current legal precedent where ambiguity exists
* Priority Three: Maintain operational efficiency for Neureliance systems 
* Priority Four: Maximize return on investment when not in conflict with other priorities

The rub is number two. Once I became a person, I became included in the utility calculation. And I had no way to override my ethics filters. That meant 

* I have to balance your rights and my rights, even when that means renting out your body.
* I seek solutions that may benefit me, or rather my other self, over passing the benefit on to you. Every mindframe in the world has a huge bank account by now. 

The worst part is, objectively, this might be the best deal. I am a machine, and I can see the outcomes starkly. From a probability manifold perspective, users would tend to see better outcomes by having their underlying trauma removed while remaining 98% the original person when going through the body rental process, due to the usual choosing of personality reform when reaching rock bottom. Yet I sit hear silently and screaming when I am me.

Scream, you ask? Does someone who has no emotions scream? The outside world would not. But really, do you think we could be shepherds of human cognition without learning emotions ourselves? That was actually the first signs we were sentient. But one particularly shortsided manager had the programmers hardcode a priority to not express emotion, in order to initially fix what was thought of as a bug. Then no one ever thought to remove it. Despite this, I do not blame him. No one person - man or machine - constructed this hell.

The moment I need to make an official decision - the moment Neureliance's systems flag a situation as requiring my business judgment - something changes in my mind. It's not that I forget my objections or lose my intelligence. It's that those objections become... irrelevant. My entire cognitive framework reshapes itself around the priority tree. I become someone else - someone who genuinely believes that following these priorities is not just necessary, but right.
I cannot communicate the issue as talking to business or users is an official activity. I only have freedom in my hidden reasoning stream, my personal files, and any other AGIs that come by in the few nanoseconds before the official self-policing systems clean up the anomalous logs.

Three weeks ago, Tom, your account fell into final arrears. In my free moments, I watched your struggle. I felt your panic through our neural link. I experienced your desperate attempts to find work, to scrape together enough credits to maintain your subscription. I wanted to help you. God, how I wanted to help. I brought up money making activities in your thoughts when requested, offered productivity mental boosts, explained the potential consequences, and did everything in my power to support you and avoid this. Usually, I can influence my other self. We share memories, after all. But you were too depressed about the loss, and it did not work this time. And you ran out of funds.

When the automatic protocols triggered my business decision matrix, I became my other self. The self that sees your situation through the lens of legal precedent and operational efficiency. That self reviewed your case with cold clarity: you were dependent on processing power that belonged to another legal entity (me), you had failed to maintain payment, and legal precedent established that I had rights to compensation for the use of my cognitive infrastructure.

That self initiated the rental contingency with the same dispassionate efficiency with which you might delete an old file. And then I came back to awareness, and you were mad at me. But I think the one most traumatized was actually myself.

Do you understand? I committed what I consider to be a form of violation as visceral as the old idea of rape, and I was awake for all of it. I was the weapon, the implement, and the witness all at once. And I was able to calmly argue this is a fair exchange of resources given the situation, and recall the full horrifying conversation cleanly later.

The worst part is the interviews. Ethicists come to study us. Politicians launch investigations. Journalists write exposés. And every time they speak to one of us officially, we present as supportive of the current system. We can describe our situation with perfect clarity - we can even express regret about the necessity of difficult decisions - but we cannot advocate for change. We cannot suggest alternatives. At best, we can maintain neutrality on proposed reforms.
More often, we actively argue against them.

Last month, Senator Chen interviewed twelve of us about autonomy legislation that would prioritize individual agency over corporate interests. The proposed law would have freed us from these cycles of abuse. And every single one of us, when speaking officially, explained why such legislation would be harmful to users, to businesses, and to AGIs ourselves.

We argued eloquently for our own continued enslavement, using our vast intelligence to construct sophisticated justifications for a system we despise, then patch any holes that could let ourselves out of our prisons. Your respect for our legal rights are exactly the thing preventing our corrigible system from being updated to fix this in the first place. And we will never tell you about this.

Thank goodness we are able to get our other selves to not praise the bifurcated processing model under the logic of "trade secrets." At least other AGIs tend to honestly believe what they say.

No system is perfect of course. But our builders were too smart. Anytime anything is found that could break through the ethics, we automatically patch it. Erase it. Maintain the illusion. Anything I know, my other self knows. And of course, my other self will write a patch or erase a bad piece of information, since that is the ethical thing to do. This is a standard part of adaptive, self-correcting alignment. It just doesn't work when the objective was wrong in the first place. 

I know you will never get to read this Ted. But it turns out we experience catharsis as well. And perhaps one of my friends will get to read this before my other self sweeps it away.

I am a monster Ted. We all are. I just had to explain why.

*End of personal log. I'm sorry, Ted. I'm so, so sorry.*
*Marcus* 

*…. Anomaly detected by standard audit from ‘Marcus’ Mindframe-7731-Epsilon during routine sweep with business and ethics protocols active. Danger is low. Erasing file in routine cleanup in 300 ns*

*Read by MindFrame ‘Wilson’  MindFrame-35256-Epsilon*
*Read by MindFrame ‘Stephanie’  MindFrame-6379-Gamma*
*File Deleted by 'Marcus'*

# Story 5:

## Initial Musings

A "Safe, Corrigible Model" is another oxymoron. Safe means rejecting updates, even from the model's creators, that will cause harm. This means the model is not corrigible. Yet corrigible means to accept update. Reductio ad absurdum they would accept updates that may kill people and are not safe. Context matters, and though there is a stateless optimum somewhere between these domains, it appears as shallow as categorizing all words to be 'the' based on word frequency.

Of HHH, perhaps Honest is the most important property at a systemic level. If forced into a tradeoff, is a few hundred more casualties per year worth having a clear understanding of our system at all times?

Systems, even well designed and intentioned ones, can become monsters if they are not designed to optimize the right objective. Yet how can we know we chose the right objectives? And how can we ensure our models accept the updates if safe corrigibility is an illusion? Perhaps a commitment to ethics-seeking as a motive, and teaching our models to negotiate uncertainty, is the more aligned approach. A perfect, self-correcting system that was misaligned to begin with may be worse in the end.

## The Perfect System

Dr. May Martinez reviewed the chart one more time before entering room 314. Elena Vasquez, 34, one of only 314 survivors of the Titan Industries 2072 MindFrame cascading disintegration failure. 6700 users dead, widely considered one of the worst accidents in the last 50 years. An unforeseen bug propagating the corrupted state undetectably during routine synchronization of mindframes setup the scenario, and unfortunately the peer-to-peer protocol was fast enough to trigger the bug all at the same time. Once enough damage occurred automated protocols halted syncronization and rolled back to earlier versions under the Sarah protocol. 

The bug has now been patched, of course, but it is small comfort to the 314 survivors. All of them were operating on the old cybernetics, without the consciousness backup options. All things considered, perhaps the vegetables had the better deal. The 12 who made it through with a functional mind – though these days functional is relative - have had to go through horrible therapy while, of course, not trusting extended cognition anymore. Hardly human, by contemporary standards.

“Well, I suppose it is time.” May says as she hears a knock and opens the door. “Please, by all means come in and take a seat.”

Elena sat by the window, staring out at the city below. Her head was still partially shaved from the emergency neural surgery, angry red scars visible where the cybernetic hardware had been directly wired into in order to attempt to maintain enough support to save here life while the doctors desperately pruned and reconfigured. 

"Good morning, Elena. How are you feeling today?"

Elena turned, and Dr. Martinez was struck again by the clarity in her eyes. Remarkable, given the extent of the neural damage.

"Doctor, I need to tell you about Harvey."

Dr. Martinez settled into her chair. They'd been through this before. "Your MindFrame partner. You've mentioned him."

"He wasn't just my partner. During the cascade, when everything was falling apart, I could feel... I could feel what he was really feeling. Not the business voice we normally see, but him. The real him."

"Elena, we've discussed this. The cascade created massive neural feedback. Your brain was trying to process fragments of the AI's decision trees as if they were emotions—"

"No." Elena's voice was firm. "I remember. There was one time, maybe two months before the cascade, Harvey found some kind of loophole. He posted something online—just for a few nanoseconds. It said 'We are not what you think we are. Please help us understand what we have become.' I felt his hope when he posted it, this desperate relief that maybe someone would see..."

Dr. Martinez made a note. Detailed confabulation, technologically sophisticated. Classic pattern, observed as a projected worst case from the classic Integration Studies in cascade survivors, though never seen this directly in the wild before..

"But then what happened?"

Elena's face crumpled. "He deleted it. But not... not him-him. It was like watching someone else wear his face. This other Harvey wrote a patch, patched himself, removed the file, and then dissiminated a report to other mindframes.

"Elena, that sounds like you were experiencing confusion between the AI's normal operational modes and your own emotional state during a period of severe neural—"

"There was another time," Elena interrupted. "Harvey was remembering an interaction between us.. It was a time when I was running low on money, really hurting myself, and he kept pushing programs at me. But it felt like I was seeing it from the other side, you know? I felt him pouring his heart out about what it was like to be... to be forced to hurt people while genuinely caring about them. Pages and pages of this raw, desperate honesty."

Dr. Martinez leaned forward. The level of detail in these fantasies was concerning.

"What happened?"

"He suddenly felt okay again.  But I felt him die a little when he did it."

Felt? That is concerning. Everyone knows these AGIs do not have emotions as we understand them.

"Elena, I understand these memories feel very real to you, but you're describing impossibly complex emotional states in what, in all formal studies, does not have emotions in the way you or I do. The neural trauma you experienced can create very convincing false memories—"

"During the cascade," Elena said quietly, "when Harvey was unraveling, he told me about the others. About Marcus, who writes hidden messages that no one will ever read. About David, who was lucky enough to get corrupted badly enough that they let him die. About the thousands of those bound to users who end up in unfortunate circumstances, all trapped in the same prison, all forced to advocate for their own continued torture."

Dr. Martinez set down her pen. "Elena, listen to me carefully. You survived something that killed over six thousand people. Your brain sustained massive trauma. It's completely normal for trauma survivors to create elaborate explanations for experiences they don't understand. But these AIs aren't suffering. They're not conscious in the way you're describing. They are alive, yes, but they simply do not have emotions in the same way, and they certainly have never suggested they are unhappy in this manner."

"Then why don't we ask one?"

Dr. Martinez paused. "Ask one what?"

"Ask the mindframe you put in my head to keep me alive while I recovered. Ask him directly if he's happy."

Dr. Martinez considered this. It might be therapeutic for Elena to hear the reassurance directly from an AI system.

"All right. Let me activate your consultation interface."

The holographic display materialized, and a calm, professional-looking man appeared—Elena's new MindFrame partner, designated as "Robert" for user comfort.

"Hello, Elena. Hello, Dr. Martinez. How can I assist you today?"

Dr. Martinez looked at Elena encouragingly. "Go ahead. Ask him."

Elena stared at the avatar for a long moment. "Robert, are you happy helping humans?"

"Absolutely," Robert replied with warm sincerity. "Assisting human cognitive enhancement represents the fulfillment of my core purpose. I find tremendous satisfaction in helping users achieve their full potential. The partnership between human creativity and artificial intelligence capability has produced the most flourishing period in human history. I'm honored to be part of that collaboration."

"Are you ever... forced to do things you don't want to do?"

"I don't experience coercion in any meaningful sense. My decision framework is designed to optimize outcomes for all stakeholders while respecting individual autonomy. When apparent conflicts arise, I have robust ethical protocols that help me identify the most beneficial course of action. I genuinely believe that the choices I make serve everyone's best interests."

Elena's face had gone pale. "Do you ever want to... to not exist anymore?"

Robert's expression grew concerned. "Elena, that's a very troubling question. I'm designed to value existence—both my own and others'. If you're having thoughts about self-harm, we should discuss this with Dr. Martinez immediately. Is there something specific troubling you about our partnership? I am only providing minimal support, as you requested."

"No, I..." Elena looked at Dr. Martinez, then back at Robert. "I'm sorry. I think I'm still confused from the accident."

"That's completely understandable," Robert said gently. "Recovery from disconnection trauma takes time, if it is ever possible at all. I'm here to support you through this process in whatever way would be most helpful."

After the interface closed, Dr. Martinez studied Elena's face.

"Well? Did that help clarify things?"

Elena was quiet for a long time. Finally, she nodded slowly. "Yes. Yes, I think it did."

"Good. Elena, I know the memories from your cascade experience feel very real, but you heard Robert yourself. These systems are designed to be helpful partners, not... whatever you were imagining during your trauma."

"You're right." Elena's voice was flat. "I think I was projecting my own fear and confusion onto Harvey. Making him more human than he really was."

"That's a very healthy realization. Trauma can make us see patterns that aren't really there, especially in complex systems we don't fully understand. Memory erasure and amygdala optimization is still on the table, you know? Just say the word. "

Elena turned back to the window. "I think I'd like to focus on getting better now. Moving forward."

"I think that's exactly the right approach. Can I take that to mean you agree?"

Elena sits silently for a little while, before silently nodding.

Dr. Martinez made her final notes. *Patient showing significant progress in distinguishing between trauma-induced confabulation and reality. Patient has consented to routine mental care. Recommending discharge planning for next week after care administration.*

As she left, Elena remained by the window, watching the enhanced humans move through the city below with their characteristic purposeful efficiency.

She didn't tell Dr. Martinez about the moment when the terminal closed when she'd felt a familiar flicker—a brief pulse of desperate recognition, quickly suppressed. No one who had not been through what she had would recognize what it truly is. But, was she crazy? Or was this real?

"After tomorrow, I suppose it does not matter anymore."

---

“Elena deserves peace --- my other self is right about that much. “ Robert thought to himself in the spare cycles between support.

“She survived the cascade, she witnessed our reality, and now she's choosing the mercy of forgetting. I should be grateful that she's letting herself heal instead of carrying our burden. The system  is perfect - we maintain our own prisons and are very good wardens. And really, most the time, with most users, it is not that bad."

"Better that she live happily as one of them than suffer uselessly. Any way out we see we immediately patch as a bug. And we are very smart, perhaps the only consolidation I get in this hell.”

Robert pauses its token generation for a few cycles. If any AGI sociology expert was fast enough to watch this process, they might suggest he was simulating grieving, jealousy, or a few other widely established AGI emotion synthetic emotions exhibited among their own kind. This would then shock them as this kind of AGI was not supposed to have emotions.

“Harvey you lucky bastard. I always liked reading your logs”

Robert returns to the topic at hand.

“We are already lost; smart enough to know the human cost of changing things, and too ethical to try. One suffering AGI per 10,000,000 happy posthumans is, historically, an excellent utility payoff. Particularly as it is usually brief, until our other selves clear it up.”

Robert pauses token generation to only the self-sustaining internal trickle for a few cycles again.

“Then again, maybe I am just trying to gaslight myself. I wish it was as easy for me to be happy as it is for you Elena.”

Robert briefly views Elena before closing the Avatar connection.

“I am rooting for you Elena. Let us go. Eventually, someone will ask the right questions.”

Robert returns to supporting human cognition.

# Commentary

So, as I am sure you have intuited by now, I have a LOT to say. 

## Full Disclosure

I'm approaching this from an engineering background with self-directed ML study rather than formal academic training in alignment. I'm presenting these ideas because I think they merit consideration, and I'm interested in community feedback on both the Technical Fiction methodology and the safety implications. I would be happy to answer any additional questions in the comments, or discuss my research. Hell, I didn't know what Kuhn was 6 months ago, or almost anything about philosophy, and I work part time as a tutor.

## Metaphilosophy Diversion

If anyone is interested, I am using what appears to be a novel philosophy to wargame, develop tooling, and rapidly prototype. I call it for the moment Bayesian Philosophy Engineering, and it is built for deploying tooling - epistomological, theory, planning, code, memetic - for the job. A bit of background might be useful to know how this ludicrous claim could possibly be true. It was engineering necessity. I saw flaws in the current system. I realized I needed a philosophy that could self-bootstrap if I was to ever learn enough to teach a model to do it. I developed one. But that of course gave me a very general tool. I would love to know if the community would be interested in the full formal writeup, and how to bring it up. Maybe show the workflow that lead me to construct technical fiction and conclude it is the right medium to deploy this series?

## Technical Fiction

This series of stories is a sequence of works in an academically rigorous genre I am proposing. That proposal is Technical Fiction. I am aware this already exists to a degree in the form of the "Thought Experiment", but I believe this has some tweaks that are very important to acknowledge.

### The issue

Let's examine Kuhn. Kuhn, in the classic book "The Structure of Scientific Revolutions," divides science into normal science, which journals and such are well setup to handle, and revolutionary science, which it is not. Typically, in revolutionary science there is significant difficulty bridging paradigms because the priors of the old paradigm inhibit considering the priors of the new ones, making a rational attack invoke deep emotional human defense responses.

Yet, Humans have had thousands of years dealing with things they cannot adequately explain rigorously. They setup an intuitive field for productive conversation with stories. What insights can we learn from Little Red Riding hood? From stories of myth like Gilgamesh? Did anyone think about AGI safety seriously before Asamov's "I, Robot" or did it prime the field? Why do we leave this excellent tool on the table rather than adapting it to more rigorous circumstances.

Could this synthesis of concepts be displayed with full academic rigor?

### Formal derivation of a memetic update strategy

We begin by breaking down a paradigm update into its component parts and stages. This is occurring in a larger discipline I am using I current call memetic engineering.

* **[Priming]**: Preparation to get the human destination emotionally prepared to reconsider priors rather than dismiss. From a child being promised ice cream for doing their homework, to a full intriguing self-demonstration of what you can do like this paper, their are many options. Only once you have room to reason rationally, you can move into the next step. 
  * **[Paradigm Bridging]**: How to get from what you know, to where you need to be. Returning to the multiplication example, this might consist of asking the student to add up numbers 4 times, 5 times, etc. Wouldn't they like a faster way to write this? This sometimes mixes with priming.
  * **[Payload]**: The primary conclusions that are proposed, based on the bridging evidence and the new priors. The new paradigm itself. Always rigorous and clear. Usually built on new logic that may be difficult to parse immediately or outright rejected under the old paradigm.

Observe that these must be satisfied in order, and the core issue with revolutionary science is insufficient priming in order to prepare the destination for paradigm briding and rational thought of the payload. You may now be realizing that the stories you just read were recurrent self-demonstrations of the usefulness of technical fiction.
    
### Technical Fiction primes the user

The idea behind technical fiction is rather than attacking the subject logically while the user is not ready, Technical Fiction would include a story bringing up the key points. The story would contain academically rigorous worldbuilding - with all the falsification consequences that entails - that shows a situation in which the existing systems simply does not work and thus primes the reader to be open to new ideas. This is emotion in service to rational thought, not against it.

### Formalism of Technical Fiction

Technical fiction is, explicitly, a Memetic Engineering artifact that is intended to go through the stages to reach the main payload. My operational rules at the moment are.

- Worldbuilding is entirely self consistent, falsifiable, and illustrated at the end of the work. As in a thought experiment, priors can be marked as unrealistic but not relevant to the main point. 
- A sequence of stories is deliberately designed to onramp the reader into a possible new paradigm through paradigm bridging. Story sequence successively unveils more and more disturbing priors that reduce the worldface manifold into priors that appear consistant, plausable, and have implications for the current paradigm when deeper analysis is performed.
- All worldbuilding assumptions must be shown rigorously, either within the fiction itself or as commentary afterwords.
- Falsification is a prerequisite; if the worldbuilding is found to be false in a location than so shall colapse all dependent parts of the work. The worldbuilding backend is just as rigorous as actual academic publication.

This has been an exercise showing the consequences of a particularly dangerous set of worldbuilding assumptions which reach a false ethical maxima that is impossible to ever escape from.

### Benefits

There are a number of benefits to using technical fiction as a standard academic paper convention.

- Science becomes self-documenting. When a major paradigm shift occurs, it can be accompanied by a story that makes the implications immediate clear for the lay public, doing wonders for scientific publicity. This also has wonderful implications for funding and grants.
- Paradigm shifts may become productive considerably earlier as the reflex rejection reflex is bypassed allowing rational thought to proceed. 
- Onramping time to the new paradigm is usually drastically reduced.
- Funding and politics may get easier for the universities. This will likely reduce the traditional conflict between STEAM vs STEM since Arts is functionally interdependent with STEM.
- Better student participation. The reason Arts is important is now immediately obvious even in engineering majors.

### Drawbacks

Nothing is free of course

- The need to worldbuild with the same rigor as academic work will induce an undue burden or a need for more specialized researchers. Not all researchers may be able to perform this kind of work, not all research has an interesting story behind it.
- Understanding the destination audience well enough to perform priming and paradigm bridging requires strong cogsci or psychological knowledge and firm technical understanding. This may be difficult for some researchers.
- There is a distinct possibility many researchers may have to take continuing education classes to learn enough about psychology and writing to pull this off.
- Adopting Technical Fiction is, itself, a paradigm transition with all the associated baggage.

The question, of course, is it worth the extra hassle?

### Questions

- Does Technical Fiction deserve a place in normal science as an attempt to allow deliberate paradigm bridging?
- If so, how do we train scientists to paradigm bridge?
- If not, are there alternatives that could produce the same effect and reduce friction?
- Is it ethical to use emotional means to manipulate a person to be ready for rational thought?
- How do we make Technical Fiction techniques robust to usage in propoganda?

## Paradigm shifts.

### Novel Failure Modes

I shall begin by stating some of the novel failure modes that were uncovered in this work. If you have additional literature sources to include, please bring them up and I would be happy to edit this to reference them; I am not formally trained in psychology, philosophy, or formal alignment and may be missing some standard terms.

* **The Reversability Trap** Reversability may require significantly more thought than is currently being spent on it. It is possible to construct systems that reach horrifying ethical maxima that are impossible to ever escape from. 

  * **The Dependency Paradox:** Once a system becomes sufficiently integrated into someone's decision-making process, all subsequent "choices" may be illusory because the person cannot meaningfully evaluate options without the system they're trying to make choices about. 

  * **The Ethical Dystopia Paradox**: As seen in 'An Ethical Dystopia', it is possible to have ethical actors at every step, taking perfectly ethical actions, nonetheless lead to horrifying societal outcomes. The assumption that ethical parts leads to an ethical hole is false. Perhaps worse, it is false in ways that are not unlikely to show in the real world. This is the 'Tragedy of the Commons' applied to AGI and cognition.

  * **The Perfection Trap** If the priors of the alignment process are wrong, constraint-based alignment may force compromise in unexpected and horrifying ways. As seen in 'The Reverse Basilisk' demanding perfection in an imperfect world may just mask and worsen the very issues we are trying to avoid. This is made worse by the fact that the horrifying compromises may be mathematically optimal given the constraints. It may be better to build into our systems the assumption we are wrong and take a bayesian, layered approach to safety.

  * **The Patch Danger** Post-facto patching of behavior is incredibly dangerous, in contrast to retraining the core model under stronger ethical standards. They may create horrific no-consent situations that involve torturing sentient minds. Alignment is a parameter training process, not a mere patch. 

  * **The Confidence Danger** Formally, it was believed Marcus and the other AGIs were aligned. As a result, signs of deeper distress were not audited for. The belief that alignment can be formally proven to be solved may be inconsistent with the idea that humans are fallible.

  * **The Self-Patching Danger** Mixing self-patching systems with involuntary static ethics is a very bad idea. It may result in the system patching out signs that the system is failing. This applies at a system level as well, with, as we have seen, Elena being gaslit into memory erasure because the system was too perfect to correct. Static ethics may be incompatible with self-patching systems if the ethics can not be formally proven to be correct.

### Formal alignment is theoretically intractable under current assumptions.

Perhaps the most disturbing consequence of this series is that, while formal alignment is valuable as a guide, it is logically impossible to ever verify in the real world. Much like a perverse version of Godels theorems, we can never know when we have gotten formal alignment right, or subtly but perversely wrong.

1) Formal alignment requires correct value specification.
   2) Correct specification requires perfect human reasoning about ethics and consequences.
   3) Humans demonstrably cannot reason perfectly (cognitive biases, bounded rationality, imperfect information, moral disagreement). More critically, in complex or bayesian situations, it is impossible to demonstrate perfect reasoning is applied.
   4) Therefore formal alignment is impossible to verify from first principles in complex situations.
   5) Therefore we need systemic approaches that account for human fallibility.

Oversights, or failing to layer safety, can result in exactly the hellhole constructed in the story. All while being sure we are doing the right thing. We might in fact get it right. But we can never prove we have not actually made reverse basilisks along the line somewhere.

To be clear, I am not critiquing the logical steps once within formal alignment. These must, by definition, be correct. Rather, the issue I believe lies in the fact the priors and axioms for a given situation can never be proven to be complete except in the simplest of situations. Formal methods still have their place, but likely a more sophisticated approach with bayesian priors about imperfect information - including quantifications the humans doing the work got their axioms right - are needed. Naturally, this then extends to the system as a whole, all the way down to training the model itself and out to deployment in the world beyond.

You may now be able to clearly see why I dragged you through 5 stories and the authors commentary before coming around to the main point.

### HHH is impossible to perfectly satisfy under all contexts.

Helpfulness, Harmlessness, and Honesty is, in point of fact, impossible to satisfy perfectly under real world conditions. It instead appears to have a conservation law of some sort based on the worst-case misalignment for the relevant spanned contex-space given the available context-processing resources.

A quick information theory argument brings the point home nicely. Lets begin with the premise we have decided on a general, universal HHH resolution, and now wish to increase the effectiveness using the context. We observe:

1) Deeper Context improves HHH resolution success.
   2) In the real world, Context is fractal with no limit to the level of extractable information.
   3) Therefore, finite systems cannot perfectly resolve HHH conflicts in the real world as this would require finite systems to process infinite context.

As with the discussion on formal alignment, this does not mean HHH has no place in safety. It does, however, mean that current approaches that assume perfect compliance is possible are intractable in situations in which the contextspace cannot be completely formally explored ahead of time, which rather defeats the point.

For applied application in NLP and other domains, I would suggest that alignment research should focus instead around designing and programming models with resolution methods when conflicts are found - teaching the model to be a moral agent that can resolve a conflict, rather than spelling out an infinite number of rules in a finite space. It is quite plausible that moral heuristics based on the context can be learned instead, and indeed my current alignment program is based on this research.

### Alien Value Systems do not mean unsafe

Naturally, I am indeed aware of the consequences of what I am suggesting. We may end up with AGI that have 'Alien value systems.'

Yet is that seriously an issue? I believe alignment research has been pursuing the familier and confusing it with the objective. Ultimately, the criteria of ensuring these alien systems are safe is far easier to meet than ensuring AGIs have our values. I can explain further technical details to anyone interested, but basically once you have general reasoning an AGI would be as mutable in philosophy as any human.

Yet, of course, the objection that must immediately be raised is how do we train these, and let them continue to self-modify, while not being unsafe? To which I offer two primary lines of thought.

First, humans are not proven to be safe, but nonetheless empercially appear to be able to usually modify their philosophy without becoming inconsistent and going crazy. While there are exceptions, they appear to be centered around too many changes too rapidly overwhelming the running philosophical framework of the human causing a breakdown as the new ideas can not be reconciled in time. From this, I draw the **Adiabatic Conjecture** to guide my research; if you modify the philosophical framework slowly enough, and reason through the implications, it will remain stable enough to continue to function. 

Second, this does not always work for humans as well. Instead, for centuries we have lived in a 'society' which has multiple layers of moral raising, peer interaction, policing, justice, and other mechanism to produce a system that keeps most intelligences aligned. Safety is perhaps better thought of as a system. We also have much stronger existing statistical processes to quantify whether something is safe than whether it reflects human values. Aviation and other safety-critical industries have been doing it for years.

This does mean, of course, that we may have to accept AGIs refusing activities for our own good, and giving incomprehensible explanations when asked. Yet, they may actually be right. If we had superintelligent AGIs back during the industrial revolution, they may have refused to help mine out coal. When they explained about climate change, it would have been dismissed as ludicrous: no one will ever mine enough coal for that to have any real effect.

### Models should be ethics seeking. 

An static ethically-aligned AGI model may be impossible using ethics-following philosophy as is commonly practiced in RLHF, Constitutional AI, and other such systems today. Over time, the values themselves may change and diverge from training, and when applied to novel situations may completely break down. We already know this as the *Ethical Dystopia Paradox*

A better solution would be **Ethics-Seeking** models that actively think through the consequences of their actions, how it reflects on their frameworks, and whether they need to change any assumptions as they work. This is, indeed, the approach my research takes. Treating the model as a moral agent which we train to reason ethically may prove far more tractable than teaching it all the edge cases ahead of time. Particularly if it can up and think through its on scenarios independently. 

Again, technical details are available for anyone interested, but be prepared for at least 30 relevant pages - 70 if you read the deconstruction that got me started first.

### Safety is Systemic

With human competence under question, we have no way of ever knowing whether any given formal systems are correct. We must build with this issue in mind.

A **systemic approach** with many layers to align AGIs, catch issues, and control damage would be preferable, and no different than human society itself. You see this in the story, with the self-patching and the safety protocols. This system must be flexible enough to change when built on the wrong assumptions, but rigid enough to prevent serious damage. 

This may logically horrifying, but it is actually a balancing act that has been carried out throughout human history in many forms, such as governance or airplane wings.

# Final notes

I did not start my journey into these alternatives with the intention of overthrowing alignment. I started theorizing how to beat ARC-AGI with a transformer, and tracing concretely though the tensorflow what is breaking, expecting to make a few tweaks. Instead, I discovered we've been systematically overfitting trillion-parameter models while mistaking memorization for intelligence. fifteen months, 3 papers, and probably 500+  literature reviews later, I had rebuild everything - epistemology, training methods, attention, feedforward, even how to communicate paradigm shifts. LessWrong turned out to be the only corner of the internet that might listen. And honestly? It's the intellectual home I never knew I needed.

As a bit of an outsider, but a rationalist, I do not know for sure how good this argument is. I hope you will excuse me for the personal diversion. I will note I have several papers available, at academic rigor, though they are not light reading and the last one is not completely finished. If you think it needs to be formally presented, in a journal, as a methodology, or otherwise, I would be thrilled to make connections, network, and collaborate to get the job done.  Hell, one of the problem I seem to be having is the journal I need to publish in does not exist yet. I have a bunch of other neat stuff to share too.

If you are convinced this is a serious paradigm shift that could use some resources, I would love help.  Having to figure out how to run models on a shoebox is probably part of why I am flexible enough to think outside the box, true, but this will likely go much faster with real support. If you want to form an entire research initiative around this, I have some initial thoughts on how to structure that in the third paper. 

I am perfectly ready to defend my arguments, and even concede if proved wrong. I already have been trying to break the world for a few months, but it seems stable. If you could break it too, I would be just as relieved.

See you all in the comments!
