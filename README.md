# operations-in-the-information-environment
## Game Scenario

There are four teams involved: Red, Blue, Green and Grey.
The scenario has been deliberately designed to represent the uneven playing field of the contested environment between the various teams. The scenario highlights the vulnerabilities of blue team in the contested information environment. The concept of blue and red teams is prevalent in cybersecurity related serious games or wargames.We are modelling the information environment in a country.
<br />
Red and Blue teams are the major geopolitical players in this fictitious country.
<br />

Red team is seeking geopolitical influence over Blue team. Of particular interest to Red team is influence over Green population and the Government. Blue is seeking to resist the Red teams growing influence in the country, and promote democratic government in the Green country.
<br />
A key challenge faced by the Blue team, that will become apparent in the exercise, is that their democratic values are leveraged against them. They are vulnerable to some forms of manipulation, yet their rules-of-engagement do not allow them to respond in equal measure: there are key limitations in the ways in which they respond and engage in this unique battlespace. The Blue team is bound by legal and ethical restraints such as free media, freedom of expression, freedom of speech.
<br />
The Green team lacks a diverse media sector, it is confused and there is a wide range of foreign news broadcasting agencies Green’s population has subscribed to. The Green population suffers from poor internet literacy, and the internet literacy can be modelled via pareto distribution. The government lacks resources to launch a decisive response to foreign influence operations and a lack of capability to discover, track and disrupt foreign influence activity.
<br />
The Red team, an authoritarian state actor, has a range of instruments, tactics and techniques in its arsenal to run influence operations. The Green government can block websites and social media platforms and censor news coverage to its domestic population whilst maintaining the capability to run sophisticated foreign influence operations through social media.
<br />
The Grey team constitutes foreign actors and their loyalties are not known.Election day is approaching and the Red team wants to keep people from voting.

## Agent Description
Reinforcement learning is a subtype of artificial intelligence which is based on the idea that a computer learn as humans do — through trial and error. It aims for computers to learn and improve from experience rather than being explicitly instructed.
<br />
In order for the computer to do this the Learning algorithms are mathematical tools implemented by the programmer which allow the agent to effectively conduct trial and error when performing a task. Learning algorithms interpret the rewards and punishments returned to the agent from the environment and use the feedback to improve the agent’s choices for the future.
<br />
In reinforcement learning, the agent faces a dilemma which is known as the exploration-exploitation tradeoff. At what point should the agent exploit options which the agent thinks to be the best rather than exploring options which have the potential to be better or worse (or vice-versa)?
<br />
This tradeoff plays into something known as the multi-armed bandit problem, which is how one should dedicate a fixed amount of resources to several different options when you can never be certain what will come of exploring each state(environment) 
<br />
Therefore we decided to use an epsilon-greedy approach which selects the action with the highest estimated reward most of the time. The aim is to have a balance between exploration and exploitation. Exploration allows us to have some room for trying new things, sometimes contradicting what we have already learned. Using this learning algorithm, our agent can converge to the optimal strategy for whatever situation it’s trying to learn
