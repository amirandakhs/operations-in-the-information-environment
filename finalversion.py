import os
import sys
import math
import mesa
import random
import inquirer
import networkx as nx
import matplotlib.pyplot as plt
from colorama import Fore, Style
import numpy as np
import json
import copy

sys.path.append(os.path.realpath(".")) # for the Inquirer library 

VOTING = 1
NOT_VOTING = 0

# SET YOURSELF IF YOU DON'T WANT TO RUN THE openGame() FUNCTION TO PLAY #
    # defaults for stuff that is going to be reset in openGame()
global isBlueAI
global isRedAI

isBlueAI = True
isRedAI = False

global blueType
global redType
blueType = 'smart'
redType = 'random'

global RED_CERTAINTY
global BLUE_CERTAINTY
RED_CERTAINTY = -1.0
BLUE_CERTAINTY = 1.0

global GREEN_COUNT
GREEN_COUNT = 30

# RUN GAME VARIABLES, GLOBAL TO PASSABLE (here for reference)
global model
global voteCount
global electionDay
global followerList
global lossScaler
global BarabasiProbability
potentList = [int,0.1,0.125,0.15,0.175] #msg potencies
lossScaler = 4
BarabasiProbability = 2

# AI VARIABLES
ROUND = 1

FIRST_MOVE = True
LAST_STATE_RED = f"voting,follower,potent_mesage,max_reward"
LAST_STATE_BLUE = f"voting,follower,potent_mesage,max_reward"

def saving_state_red(state):
    """
        This function will save the resulting state into the state function
    """

    with open('state.json', 'w') as json_file:
        json.dump(state, json_file)

def read_state_red():
    """
    This function will load the state and return python dictionary
    """
    with open('state.json') as json_file:
        data = json.load(json_file)    
    
    return data

def checking_moves_red(redAgent, model):
    """
    This will make all the possible choices and return the action that return 
    the best result
    """ 
    global GREEN_COUNT
    global electionDay
    global followerList
    global potentList
    global RED_CERTAINTY 
    # Creating the log file
    file_object = open('log_red.txt', 'a')   
    
    # result with the 0 potency message 
    voting = checkOpinion()
    follower = redAgent.followers
    notVoting = GREEN_COUNT-voting
    max_reward = notVoting + follower
    potent_mesage = 0
    file_object.write(f'in the round {electionDay} with potenty message {0} -> state score {max_reward} = not voting {notVoting} + follower num {follower} \n')

    for j in range(1, len(potentList)):
        # temp model to keep the orginal mode
        # temp follower and vote count to get the temp reward

        tempModel = copy.deepcopy(model)
        tempFollower = copy.deepcopy(followerList)
        agent = copy.deepcopy(redAgent)
        voteCount = 0
        
        # update their values according to the potency
        for i in tempModel.schedule.agents:
            newVal = 0
            
            # skip node if not subscribed to red
            if tempFollower[i.unique_id] == False: 
                continue

            # calculates the new red value
            newVal = i.uncertainty - math.exp(abs(RED_CERTAINTY) - abs(i.uncertainty)) / abs(RED_CERTAINTY) * potentList[j]

            # creates the probabilty of losing this node as a follower
            lossProbability = (max(newVal,i.uncertainty) - min(newVal,i.uncertainty))   # calculate the difference between new and old certainties
            lossProbability = round( lossProbability/lossScaler, 4)                              # round to 4dp, divide to de-strengthen the probability
            spaceSize = abs(RED_CERTAINTY - BLUE_CERTAINTY)

            # weighted probability for unsubscribing, bigger change is bigger probability
            decision = random.choices([True, False], [lossProbability, spaceSize-lossProbability], k=1)

            # unsubscribe from red media, reset the uncertainty, switch opinion to blue
            if decision[0] == True: 
                # print(Fore.GREEN + 'Lost Follower, ID: '+ str(i.unique_id)) #flag
                tempFollower[i.unique_id] = False
                agent.followers -= 1
                if i.uncertainty <= 0:
                    i.uncertainty = 0.00
                    i.opinion = VOTING
                continue

            if i.uncertainty > 0:
                i.opinion = VOTING
                voteCount += 1
            elif i.uncertainty == 0:
                if i.opinion == VOTING:
                    voteCount += 1
        
        reward = (GREEN_COUNT- voteCount) + agent.followers
        if reward > max_reward:
             
            max_reward = reward
            potent_mesage = j 
        # Adding some randomness to our agent when the situation is equal
        elif reward == max_reward:
            chance = random.choice([True,False])
            if chance:
                max_reward = reward
                potent_mesage = j 
        file_object.write(f'in the round {electionDay} with potenty message {j} -> state score {reward} = not voting {(GREEN_COUNT- voteCount)} + follower num {agent.followers} \n')
    
    # print('n\RED FOLLOWER COUNT: '+str(redAgent.followers)+', OUT OF: '+str(redAgent.initialFollowing)+'\n') #flag
    # closing the log file

    file_object.write(f'ROUND {electionDay} potenty message {potent_mesage} with max reward of {max_reward} has been choosen \n ')
    file_object.write(f'LOCAL CHOICE HAS BEEN MADE \n')
    file_object.close()
    return (max_reward,potent_mesage)


def red_agent_move(red, model):
    """
    This function will make the best action base on the Q learning and 
    Markov model 
     Q(s,a) <= Q(s,a) + ἄ(R + & maxa(Q(s',a))-Q(s,a) )
     
    """
    global LAST_STATE_RED
    global FIRST_MOVE
    global GREEN_COUNT


    states= read_state_red()

    # checking current state
    voting = checkOpinion()
    follower = red.followers
    notVoting= GREEN_COUNT-voting
    current_state_score = notVoting + follower
    
    a = 0.5
    b = 0.8
    # checking for the first move
    if FIRST_MOVE:
        FIRST_MOVE = False
    else:
        # updating the last move actual result
        # voting,follower,potent_mesage,current_state_score
        ls = LAST_STATE_RED.split(",")
        # learning from the experience Q(actual) - Q(estimate)
        # print("=====================================")
        # print(LAST_STATE_RED)
        # print(states)
        # print("=====================================") debugging
        if states.get(f"{ls[0]},{ls[1]}") != None and states.get(f"{ls[0]},{ls[1]}").get(ls[2]) !=None:
            Qest = states[f"{ls[0]},{ls[1]}"][ls[2]]
            Qdiff = current_state_score - Qest
            # updating the state value base on the experience
            states[f"{ls[0]},{ls[1]}"][ls[2]] += a*b*Qdiff 
    
    
    #  if agent hasn't been in this state
    if states.get("f{notVoting},{follower}") == None:
        # We choose the best local possible option
        max_reward,potent_mesage = checking_moves_red(red, model)
        reward  = max_reward - current_state_score 
        # print(type(potent_mesage)) debugging
        states[f"{notVoting},{follower}"] = {potent_mesage: current_state_score + a* reward}
        LAST_STATE_RED = f"{notVoting},{follower},{potent_mesage},{current_state_score}"
        saving_state_red(states)
        return potent_mesage
    
    else:
        #Checking the best local move
        max_reward,potent_mesage = checking_moves_red(red, model)
        # checking the reward
        reward  = max_reward - current_state_score
        # this decision is totally dependent of the a. if a is high it will be more exploring agent
        # if a is low it will depend more in the experience 
        state_reward = current_state_score + a* reward
        action_state = states[f"{notVoting},{follower}"]
        result = int
        for act in action_state:
            if action_state[act] > state_reward:
                result = int(act)
        
        if result == int:
            LAST_STATE_RED = f"{notVoting},{follower},{potent_mesage},{current_state_score}"
            states[f"{notVoting},{follower}"] = {potent_mesage: state_reward}
            return potent_mesage
        else:
            LAST_STATE_RED = f"{notVoting},{follower},{result},{current_state_score}"
            return result

class RedAgent:
    # initialise the independant variables
    def __init__(self, followers):
        self.uncertainty = potentList
        self.initialFollowing = followers
        self.followers = followers

    # if red is a human player, provide a UI and input
    def humanRed(self):
        # flag stops your turn from skipping if you view the map
        viewMap = True
        while viewMap == True: 
            redQuestions = [        
                    inquirer.List(
                    'redChoice',
                    message='What would you like to do?: ', 
                    choices=['View Network','Spread Propaganda'])]
            redChoice = inquirer.prompt(redQuestions)['redChoice']  # Pick a move to make as red
            
            if redChoice == 'View Network':
                drawModel()

            elif redChoice == 'Spread Propaganda':
                potencyQuestion = [
                        inquirer.List(
                        'potency',
                        message='How strong? (0 = SKIP): ',
                        choices=['0','1','2','3','4'])]
                potency = int(inquirer.prompt(potencyQuestion)['potency'])

                self.moveRed(potency)
                viewMap = False

    def moveRed(self,p):
        # add the follower dictionary, as well as the current game
        global followerList
        global model
        global potentList
        global RED_CERTAINTY

        # 0 potency represents a skip
        if p == 0:
            return

        # update their values according to the potency
        for i in model.schedule.agents:
            newVal = 0

            # skip node if not subscribed to red
            if followerList[i.unique_id] == False:
                continue
            
            # calculates the new red value
            newVal = i.uncertainty - math.exp(abs(RED_CERTAINTY) - abs(i.uncertainty)) / abs(RED_CERTAINTY) * potentList[p]

            # creates the probabilty of losing this node as a follower
            lossProbability = (max(newVal,i.uncertainty) - min(newVal,i.uncertainty))   # calculate the difference between new and old certainties
            lossProbability = round( lossProbability/lossScaler, 4)                              # round to 4dp, divide to de-strengthen the probability
            spaceSize = abs(RED_CERTAINTY - BLUE_CERTAINTY)

            # weighted probability for unsubscribing, bigger change is bigger probability
            decision = random.choices([True, False], [lossProbability, spaceSize-lossProbability], k=1)

            # unsubscribe from red media, reset the uncertainty, switch opinion to blue
            if decision[0] == True: 
                print(Fore.GREEN + 'Lost Follower, ID: '+ str(i.unique_id))
                followerList[i.unique_id] = False
                self.followers -= 1
                if i.uncertainty <= 0.00:
                    i.uncertainty = 0.00
                    i.opinion = VOTING
                continue
            
            # round the uncertainty to 2dp
            i.uncertainty = round(mediate(newVal),2)

        #### PRINT FOR TESTING PURPOSES, DELETE BEFORE SUBMISSION ####
        # for i in model.schedule.agents:
        #     print(Fore.RED + 'ID: '+str(i.unique_id)+ ' HAS UC: '+str(i.uncertainty))
        print(Fore.GREEN + 'RED FOLLOWER COUNT: '+str(self.followers)+', OUT OF: '+str(self.initialFollowing))

    def redRandom(self):
        potentChoice = random.choices([0,1,2,3,4], [0.2,0.2,0.2,0.2,0.2], k=1)
        potentChoice = potentChoice[0]
        self.moveRed(potentChoice)

class BlueAgent:
    # initialise the blue agent
    def __init__(self, energy):
        self.initialEnergy = energy
        self.energy = energy
        self.lastGrey = 'NONE'
 
        # grey agents are hardcoded as 10% of the green population
        self.greyAgents = round(energy/10)
        if self.greyAgents <= 0:    # one agent if less than 10 population
            self.greyAgents = 1
    
    def moveGrey(self): # imitates the max payoff of its loyal team
        # add global models
        global followerList
        global model
        global potentList

        team = random.choices(['RED', 'BLUE'], [0.4, 0.6], k=1)
        
        # max potencies are used, [-1] to refer to back of potency list
        if team[0] == 'RED':
            for i in model.schedule.agents:
                if followerList[i.unique_id] == False:  # doesn't include non-subscribed greens
                    continue
                newVal = i.uncertainty - math.exp(abs(RED_CERTAINTY) - abs(i.uncertainty)) / abs(RED_CERTAINTY) * potentList[-1]
                i.uncertainty = round(mediate(newVal),2)
        
        else:
            for i in model.schedule.agents:
                newVal = i.uncertainty + math.exp(abs(BLUE_CERTAINTY) - abs(i.uncertainty)) / abs(BLUE_CERTAINTY) * potentList[-1]
                i.uncertainty = round(mediate(newVal),2)

        self.greyAgents -= 1    # remove an agent for disposal 
        self.lastGrey = team[0]

    # if blue is a human player, provide a UI and input
    def humanBlue(self):
        # flag stops your turn from skipping if you view the map
        viewMap = True
        while viewMap == True:
            blueQuestions = [
                    inquirer.List(
                    'blueChoice',
                    message='What would you like to do?: ',
                    choices=['View Network','Defend Political Ideology','Gamble(Grey)'])]
            blueChoice = inquirer.prompt(blueQuestions)['blueChoice']   # Pick a move to make as blue
            
            if blueChoice == 'View Network':
                drawModel()

            elif blueChoice == 'Defend Political Ideology':
                potencyQuestion = [
                        inquirer.List(
                        'potency',
                        message='How strong? (0 = SKIP): ',
                        choices=['0','1','2','3','4'])]
                potency = int(inquirer.prompt(potencyQuestion)['potency'])

                self.moveBlue(potency)
                viewMap = False

            elif blueChoice == 'Gamble(Grey)':
                if self.greyAgents <= 0:
                    print(Fore.BLUE + '\nYOU\'RE OUT OF GREYS, PICK A DIFFERENT MOVE')
                else:
                    self.moveGrey()
                    print(Fore.YELLOW + self.lastGrey + ' WAS CHOSEN, '+ str(self.greyAgents) + ' GREYS REMAINING')
                    viewMap = False

    def moveBlue(self,p):   # imitates the red move, besides the subscriptions
        global model
        global potentList
        global BLUE_CERTAINTY
        global GREEN_COUNT              # uses this to create a percentage of current voters
        global voteCount                # relies on this being updated prior to making a move

        if isBlueAI == True:
            if self.greyAgents > 0:
                energylvl = round(self.energy/self.initialEnergy,2)
                opinionlvl = round(voteCount/GREEN_COUNT,2)
            
                if opinionlvl < 0.5:
                    multi = 0.5 + (0.5 - opinionlvl)
                    greyOdds = round(((1-opinionlvl)*(1-energylvl) * multi),2)
                    greyChance = random.choices([True, False], [greyOdds, 1.0-greyOdds])
                    if greyChance[0] == True:
                        self.moveGrey

        # 0 potency represents a skip
        if p == 0:
            return
        
        # if you're out of energy, you can't make any moves
        if self.energy <= 0:
            print(Fore.BLUE + '\nYOU\'RE OUT OF ENERGY, YOUR TURN HAS BEEN AUTO SKIPPED')
            return

        # update their values according to the potency
        for i in model.schedule.agents:
            newVal = 0

            # calculates the new blue value
            newVal = i.uncertainty + math.exp(abs(BLUE_CERTAINTY) - abs(i.uncertainty)) / abs(BLUE_CERTAINTY) * potentList[p]
            
            # creates the probabilty of losing energy
            lossProbability = (max(newVal,i.uncertainty) - min(newVal,i.uncertainty))   # calculate the difference between new and old certainties
            lossProbability = round( lossProbability/lossScaler, 4)                              # round to 4dp, divide to de-strengthen the probability
            spaceSize = abs(RED_CERTAINTY - BLUE_CERTAINTY)

            # weighted probability for unsubscribing, bigger change is bigger probability
            decision = random.choices([True, False], [lossProbability, spaceSize-lossProbability], k=1)
            if decision[0] == True:
                self.energy -= 1

            # round the uncertainty to 2dp
            i.uncertainty = round(mediate(newVal),2)
        
        #### PRINT FOR TESTING PURPOSES, DELETE BEFORE SUBMISSION ####
        for i in model.schedule.agents:
            print(Fore.BLUE + 'ID: '+str(i.unique_id)+ ' HAS UC: '+str(i.uncertainty))
        print(Fore.GREEN + 'BLUE ENERGY COUNT: '+str(self.energy)+', OUT OF: '+str(self.initialEnergy)+'\n')

    def blueRandom(self):
        potentChoice = random.choices([0,1,2,3,4], [0.2,0.2,0.2,0.2,0.2], k=1)
        potentChoice = potentChoice[0]
        self.moveBlue(potentChoice)

def saving_state_blue(state):
    """
    This function will save the resulting state into the state function
    """

    with open('state1.json', 'w') as json_file:
        json.dump(state, json_file)



def read_state_blue():
    """
    This function will load the state and return python dictionary
    """
    with open('state1.json') as json_file:
        data = json.load(json_file)   
    return data

def checking_moves_blue(blueAgent, model):
    """
    This will make all the possible choices and return the action that return 
    the best result
    """ 
    global electionDay
    global potentList
    global BLUE_CERTAINTY

    file_object = open('log_blue.txt', 'a')  

    voting = checkOpinion()    
    energy = blueAgent.energy
    max_reward =  voting + energy
    potent_mesage = 0
    file_object.write(f'in the round {electionDay} with potenty message {0} -> state score {max_reward} = voting {voting} + energy left {energy} \n')


    for j in range(1,len(potentList)):
        # temp model to keep the orginal mode
        # temp follower and vote count to get the temp reward

        tempModel = copy.deepcopy(model)
        tempEnergy = blueAgent.energy
        voteCount = 0

        for i in tempModel.schedule.agents:
            newVal = 0

            # calculates the new blue value
            newVal = i.uncertainty + math.exp(abs(BLUE_CERTAINTY) - abs(i.uncertainty)) / abs(BLUE_CERTAINTY) * potentList[j]
            
            # creates the probabilty of losing energy
            lossProbability = (max(newVal,i.uncertainty) - min(newVal,i.uncertainty))   # calculate the difference between new and old certainties
            lossProbability = round( lossProbability/lossScaler, 4)                              # round to 4dp, divide to de-strengthen the probability
            spaceSize = abs(RED_CERTAINTY - BLUE_CERTAINTY)

            # weighted probability for unsubscribing, bigger change is bigger probability
            decision = random.choices([True, False], [lossProbability, spaceSize-lossProbability], k=1)

            if decision[0] == True:
                tempEnergy -= 1


            i.uncertainty = round(mediate(newVal),2)
            # checking the votes
            
            if i.uncertainty < 0:
                i.opinion = NOT_VOTING
            elif i.uncertainty > 0:
                i.opinion = VOTING
                voteCount += 1
            elif i.uncertainty == 0:
                if i.opinion == VOTING:
                    voteCount += 1
        
        reward = voteCount + tempEnergy
        
        if reward > max_reward:
            max_reward = reward
            potent_mesage = j
        # Adding some randomness to our agent when the situation is equal
        elif reward == max_reward:
            chance = random.choice([True,False])
            if chance:
                max_reward = reward
                potent_mesage = j 
        file_object.write(f'in the round {electionDay} with potenty message {j} -> state score {reward} = voting {voting} + energy left {tempEnergy} \n')

    
    # print('\nBLUE ENERGY COUNT: '+str(blueAgent.energy)+', OUT OF: '+str(blueAgent.initialEnergy)+'\n') # flag
    # Closing the log file
    file_object.write(f'ROUND {electionDay} potenty message {potent_mesage} with max reward of {max_reward} has been choosen \n ')
    file_object.write(f'LOCAL CHOICE HAS BEEN MADE \n')
    file_object.close()
    return (max_reward,potent_mesage)


def blue_agent_move(blue, model):
    """
    This function will make the best action base on the Q learning and 
    Markov model 
     Q(s,a) <= Q(s,a) + ἄ(R + & maxa(Q(s',a))-Q(s,a) )
    """
    global LAST_STATE_BLUE
    global FIRST_MOVE


    states = read_state_blue()
    voting = checkOpinion()

    
    energy = blue.energy
    current_state_score = voting + energy
    
    a = 0.5
    b = 0.8
    # checking for the first move
    if FIRST_MOVE:
        FIRST_MOVE = False
    else:
        # updating the last state move with actual result
        # voting,follower,potent_mesage,current_state_score
        ls = LAST_STATE_BLUE.split(",")
        # learning from the experience Q(actual) - Q(estimate)

        # print(ls)
        # print(states[f"{ls[0]},{ls[1]}"]) # debugging
        # print(states[f"{ls[0]},{ls[1]}"][ls[2]]) # debugging
        if states.get(f"{ls[0]},{ls[1]}") != None and states.get(f"{ls[0]},{ls[1]}").get(ls[2]) !=None:
            Qest = states[f"{ls[0]},{ls[1]}"][ls[2]]
            Qdiff = current_state_score - Qest
            # updating the state value base on the experience
            states[f"{ls[0]},{ls[1]}"][ls[2]] += a*b*Qdiff     


    #  if agent hasn't been in this state
    if states.get(f"{voting},{energy}") == None:
        # We choose the best local possible option
        max_reward,potent_mesage = checking_moves_blue(blue, model)
        reward  = max_reward - current_state_score 
        states[f"{voting},{energy}"] = {potent_mesage: current_state_score + a* reward}
        LAST_STATE_BLUE = f"{voting},{energy},{potent_mesage},{current_state_score}"
        saving_state_blue(states)
        return potent_mesage
    
    else:
        #Checking the best local move
        max_reward,potent_mesage = checking_moves_blue(blue, model)
        # checking the reward
        reward  = max_reward - current_state_score
        state_reward = current_state_score + a* reward
        action_state = states[f"{voting},{energy}"]
        result = int
        for act in action_state:
            if action_state[act] > state_reward:
                result = int(act)
        LAST_STATE_BLUE = f"{voting},{energy},{potent_mesage},{current_state_score}"
        if result == int:
            states[f"{voting},{energy}"] = {potent_mesage: state_reward}
            return int(potent_mesage)
        else:
            return result


class GreenAgent(mesa.Agent):
    # A green agent with associated variables

    def __init__(self, unique_id, model, opinion, uncertainty):
        super().__init__(unique_id, model)
        self.opinion = opinion
        self.uncertainty = uncertainty
        self.connected = True

    # Simulates the green interaction round, by having each node talk to its neighbour only ONCE
    def step(self): 
        neighbors_nodes = self.model.grid.get_neighbors(self.unique_id)
        np.random.shuffle(neighbors_nodes) # shuffle so they don't talk in ascending order constantly
        # print('NODE TURN: '+str(self.unique_id))
        # print('NEIGHBOURS: '+str(neighbors_nodes))
        for i in range(len(neighbors_nodes)):
            neighbor = self.model.schedule.agents[neighbors_nodes[i]]
            # print(str(neighbor.unique_id) + ' TURN')

            try:
                if moveDict[str(self.unique_id)+str(neighbor.unique_id)] == True : # If nodes have already talked previously skip
                    # print(str(self.unique_id)+ ' AND '+ str(neighbor.unique_id) + ' HAVE TALKED BEFORE')
                    continue
                elif moveDict[str(neighbor.unique_id)+str(self.unique_id)] == True: # If nodes have already talked previously skip
                    # print(str(neighbor.unique_id)+ ' AND '+ str(self.unique_id) + ' HAVE TALKED BEFORE')
                    continue
            except KeyError:
                pass

            talkProb = self.model.G.edges[neighbor.unique_id, self.unique_id]['weight']
            talkChance = random.choices([True, False],[talkProb, (100 - talkProb)], k=1)

            if talkChance == [True]:
                x = self.uncertainty
                y = neighbor.uncertainty
                x, y = simpleRule(x,y)
                self.uncertainty = x
                self.model.schedule.agents[neighbors_nodes[i]].uncertainty = y

                moveDict[str(self.unique_id)+str(neighbor.unique_id)] = True # save as a str to the dictionary
                moveDict[str(neighbor.unique_id)+str(self.unique_id)] = True # save as a str to the dictionary
        
        # for i in model.schedule.agents:
        #     print(Fore.GREEN + 'ID: '+str(i.unique_id)+', UC: '+str(i.uncertainty)+ Style.RESET_ALL)
        # print()

class GreenModel(mesa.Model):
    # A model of the Green network
    def __init__(self, N, BP):
        global RED_CERTAINTY
        global BLUE_CERTAINTY
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.G = nx.barabasi_albert_graph(n=self.num_agents, m=BP)  # BARABASI ALBERT GRAPH, PREFERRENTIAL ATTATCHMENT SIMULATION
        self.grid = mesa.space.NetworkGrid(self.G)                  # Add nodes to the grid for Mesa - this is how you access neighbours
        
        for (u, v) in self.G.edges():
            self.G.edges[u,v]['weight'] = random.randint(0,100) # weighted edge, experiment

        # Create agents
        for i, node in enumerate(self.G.nodes()):
            uncertainty = round(random.uniform(RED_CERTAINTY,BLUE_CERTAINTY),2) # REVERT - FOR TESTING

            # create their opinions, random if it's 0.00 uncertainty
            if uncertainty < 0:
                opinion = NOT_VOTING
            elif uncertainty > 0:
                opinion = VOTING
            elif uncertainty == 0:
                opinion = random.randint(0, 1)
            
            a = GreenAgent(i, self, opinion, uncertainty)   # Add the agent to the network
            self.schedule.add(a)                            # Adds to a schedule to access agents and simulate interaction
            self.grid.place_agent(a, node)                  # Place green agent on grid so you can access and see its position

    # simulates an interaction between green nodes
    def step(self):
        global moveDict
        moveDict = {}
        self.schedule.step()

# Some text for visualisation in CLI
def printLines(): 
        print()
        print(Fore.RED + '...'*10 + '\n' + Fore.WHITE + '...'*10 + '\n' + Fore.BLUE + '...'*10)
        print(Fore.WHITE + 'WELCOME TO DEFENDER, A COLOUR VOTING GAME')
        print(Fore.RED + '...'*10 + '\n' + Fore.WHITE + '...'*10 + '\n' + Fore.BLUE + '...'*10)
        print()

# Error catcher for user input on non possible green node desired population entry
def population_validation(answers, current): 
    if int(current) <= 0:
        raise inquirer.errors.ValidationError("", reason="Please enter an amount larger than 0")
    return True

# Error catcher for user input on non possible green node desired population entry
def intervalRed_validation(answers, current): 
    if float(current) >= 0:
        raise inquirer.errors.ValidationError("", reason="Please enter an amount smaller than 0")
    return True

# Error catcher for user input on non possible green node desired population entry
def intervalBlue_validation(answers, current): 
    if float(current) <= 0:
        raise inquirer.errors.ValidationError("", reason="Please enter an amount larger than 0")
    return True
    
# Draws the network on the screen for the player to see
def drawModel(): 
    global pos
    global model
    colour_map = []
    for i in model.schedule.agents:     # Colour coordinate the nodes for visual purposes
        # print(Fore.WHITE + 'ID: '+ str(i.unique_id)+ ' HAS UC: '+str(i.uncertainty)) # DELETE BEFORE SUBMISSION
        if i.opinion == VOTING:
            colour_map.append('mediumblue')
        else:
            colour_map.append('firebrick')

    nx.draw(model.G, pos=pos, node_size=90, node_color = colour_map, edge_color = 'black', alpha=0.90,font_size=14, with_labels=True)
    plt.show()

# Checks the opinion count of the current network USES GLOBAL MODEL
def checkOpinion():     
    global model
    voteCount = 0
    for i in model.schedule.agents: # if uncertainty is 0, leave as last opinion
        # print('ID: ' + str(i.unique_id)+' UC: ' + str(i.uncertainty))
        if i.uncertainty < 0:
            i.opinion = NOT_VOTING
        elif i.uncertainty > 0:
            i.opinion = VOTING
            voteCount += 1
        elif i.uncertainty == 0:
            if i.opinion == VOTING:
                voteCount += 1
    return voteCount

# Reduces the uncertainties down to 2 decimal places
def mediate(val): 
    global RED_CERTAINTY
    global BLUE_CERTAINTY
    newVal = val
    if newVal < 0:
        if newVal < RED_CERTAINTY:
            newVal = RED_CERTAINTY
    else:
        if newVal > BLUE_CERTAINTY:
            newVal = BLUE_CERTAINTY
    return newVal

# A simple voter rule that Green nodes talk to eachother through
def simpleRule(x,y): 
    if x < 0 and y > 0 or x > 0 and y < 0:
        diff = abs(x-y)
    else:
        diff = abs(x+y)

    if diff == 0:
        return(round(x,2),round(y,2))

    constant = 0.35

    xNew = x + ((y/diff)*constant)
    yNew = y + ((x/diff)*constant)

    xReturn, yReturn = mediate(xNew), mediate(yNew)
    return(round(xReturn,2),round(yReturn,2))

# Prints text during the game to indicate its reds move
def redMovePrint(followers,total):  
    print(Fore.RED + '\nRED TO MOVE')
    print(Fore.RED + '...\n'*2)
    print(Fore.GREEN + str(followers) + ' FOLLOWERS LEFT, '+str(total) + ' INITIALLY')

# Prints text during the game to indicate its blues move
def blueMovePrint(energy,total):    
    print(Fore.BLUE + '\nBLUE TO MOVE')
    print(Fore.BLUE + '...\n'*2)
    print(Fore.GREEN + str(energy) + ' ENERGY LEFT, '+str(total) + ' INITIALLY')

def greenMovePrint():
    print(Fore.CYAN + '\nGREEN TO MOVE...')
    print(Fore.CYAN + '...\n'*2)

# Takes all the desired inputs from open() and runs a game
def run(redAI, blueAI, redInt, blueInt, greenCount, BarabasiProbability):

    global voteCount    # current votes, winners are: < 50% is red and > 50% is blue 
    global electionDay  # days until election
    global model        # Accessible for all classes and functions, only one game at a time so...
    global pos          # makes the graph for the game

    global isBlueAI
    global isRedAI
    global blueType
    global redType

    global ROUND
    global FIRST_MOVE
    global LAST_STATE_BLUE
    global LAST_STATE_RED

    LAST_STATE_BLUE = f"voting,follower,potent_mesage,max_reward"
    LAST_STATE_RED = f"voting,follower,potent_mesage,max_reward"
    FIRST_MOVE = True
    ROUND = 1

    model = GreenModel(greenCount, BarabasiProbability) # Create a model agent
    pos = nx.spring_layout(model.G)                 # Creates the network state
    
    red = RedAgent(greenCount)                      # Make a new Red agent
    blue = BlueAgent(greenCount)                    # Make a new Blue agent
    voteCount = checkOpinion()                      # initialise current opinions, sets the correct opinion for each node

    global followerList                             # stores if the id of a green node is subscribed to red media
    followerList = {}                               # reset it from last game
    for i in model.schedule.agents:
        followerList[i.unique_id] = True

    electionDay = 0                                 # new election, reset the days left
    while electionDay <= 20:

        print(Fore.YELLOW + 'DAYS UNTIL ELECTION: ' + str(20-electionDay))
        redMovePrint(red.followers, red.initialFollowing)
        print(Fore.GREEN + 'CURRENT POLLS: '+str(voteCount)+', OUT OF: '+str(greenCount)+'\n' + Style.RESET_ALL)

        if redAI == False:                          # if no AI, wait for human input
            red.humanRed()
        else:
            if redInt == 'smart':
                option = red_agent_move(red, model)
                red.moveRed(option)
            else:
                red.redRandom()

        voteCount = checkOpinion()                  # correct the opinion of each green node
        blueMovePrint(blue.energy, blue.initialEnergy)
        print(Fore.GREEN + 'CURRENT POLLS: '+str(voteCount)+', OUT OF: '+str(greenCount)+'\n' + Style.RESET_ALL)
        
        if blueAI == False:                         # if no AI, wait for human input
            blue.humanBlue()
        else:
            if blueInt == 'smart':
                option = blue_agent_move(blue, model)
                blue.moveBlue(option)
            else:
                blue.blueRandom()
                
        voteCount = checkOpinion()                  # correct the opinion of each green node
        greenMovePrint()
        print(Fore.CYAN + 'CURRENT POLLS: '+str(voteCount)+', OUT OF: '+str(GREEN_COUNT)+'\n' + Style.RESET_ALL)
        model.step() # GREEN TO MOVE
        
        # for i in model.schedule.agents:
        #     print(Fore.GREEN + 'ID: '+str(i.unique_id)+ ' HAS UC: '+str(i.uncertainty)) # DELETE BEFORE SUBMISSION
        print()

        electionDay += 1
        
    if voteCount < greenCount/2:
        print('RED WINS!')
    if voteCount > greenCount/2:
        print('BLUE WINS!')
    if voteCount == greenCount/2:
        print('DRAW!')

def openGame():
    global GREEN_COUNT
    global RED_CERTAINTY
    global BLUE_CERTAINTY
    global blueType
    global redType
    global isBlueAI
    global isRedAI
    global lossScaler
    global BarabasiProbability

    printLines()
    print(Style.RESET_ALL)

    #### WHO IS AI, WHO IS HUMAN???
    questions = [
        inquirer.List(
        'PlayerCount',
        message='Please enter the number of desired players: ',
        choices=['0', '1', '2'])]
    playerCount = int(inquirer.prompt(questions)['PlayerCount'])

    isBlueAI = False
    isRedAI = False

    #### IF NO HUMANS, MAKE ALL AI
    if playerCount == 0:
        isRedAI = True
        questions = [
            inquirer.List(
            'RedType',
            message='Choose your desired AI for Red: ',
            choices=['random', 'smart'])]
        redType = str(inquirer.prompt(questions)['RedType'])

        isBlueAI = True
        questions = [
            inquirer.List(
            'BlueType',
            message='Choose your desired AI for Blue: ',
            choices=['random', 'smart'])]
        blueType = str(inquirer.prompt(questions)['BlueType'])

    #### IF ONE AI EXISTS...
    if playerCount == 1:
        questions = [
            inquirer.List(
            'whoIsAI',
            message='There is an AI at play. Which team would you like it to play as?: ',
            choices=['Red', 'Blue'])]
        whoIsAI = str(inquirer.prompt(questions)['whoIsAI'])

        questions = [
            inquirer.List(
            'randomOrSmart',
            message='Are they a random agent, or a smart agent AI?: ',
            choices=['random', 'smart'])]
        intelligence = str(inquirer.prompt(questions)['randomOrSmart'])

        if whoIsAI == 'Red':
            isRedAI = True
            redType = intelligence
        elif whoIsAI == 'Blue':
            isBlueAI = True
            blueType = intelligence

    #### HOW MANY GREEN NODES/AGENTS?
    questions = [
        inquirer.Text('Population', message='How big is the Green population?: ',
            validate=population_validation)]
    greenCount = int(inquirer.prompt(questions)['Population'])
    GREEN_COUNT = greenCount

    #### INTERVAL FOR RED UNCERTAINTY
    questions = [
        inquirer.Text('IntervalR', message='State an uncertainty interval for Red < 0 (-1 recommended): ',
            validate=intervalRed_validation)]
    intervalRed = float(inquirer.prompt(questions)['IntervalR'])
    RED_CERTAINTY = intervalRed

    #### INTERVAL FOR BLUE UNCERTAINTY
    questions = [
        inquirer.Text('IntervalB', message='State an uncertainty interval for Blue > 0 (+1 recommended): ',
            validate=intervalBlue_validation)]
    intervalBlue = float(inquirer.prompt(questions)['IntervalB'])
    BLUE_CERTAINTY = intervalBlue

    #### BARABASI MODEL - HOW MANY NEW EDGES PER NODE?
    questions = [
        inquirer.List(
        'GraphProbability',
        message='We use a Barabasi-Albert Model. State the number of edges to attach from a new node to existing nodes (2-3 suggested): ',
        choices=['1', '2', '3', '4', '5'])]
    BarabasiProbability = int(inquirer.prompt(questions)['GraphProbability'])

    #### LOSS-SCALING FOR FOLLOWERS/ENERGY 
    questions = [
        inquirer.Text('scaling', message='State your lossScaling for the game. Bigger number = less affect. Default for (-1&1) intervals recommend is 1.25: ',
            validate=intervalBlue_validation)]
    chosenScale = float(inquirer.prompt(questions)['scaling'])
    lossScaler = chosenScale

    print('\nOkay, let\'s begin!...')

    run(isRedAI, isBlueAI, redType, blueType, GREEN_COUNT, BarabasiProbability)
    print('Game finished!\n')
    print('Thanks for playing!\n')

# run(isRedAI, isBlueAI, redType, blueType, GREEN_COUNT, BarabasiProbability)
openGame()