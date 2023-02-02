import numpy as np
state = {'obs':([[[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

       [[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

       [[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

       [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

       [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]), 'legal_actions': [308, 4, 5, 7, 9, 11, 12]}
NumtoCard = {0:'3',1:'4',2:'5',3:'6',4:'7',5:'8',6:'9',7:'T',8:'J',9:'Q',10:'K',11:'A',12:'2',13:'B',14:'R'}

cardsIndex = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '10': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11,'2': 12, '*': 13, '$': 14}

def numpytostr(state):
    handcard = np.array(state["obs"][0])
    one_handcard = np.array(state["obs"][2])
    two_handcard = np.array(state["obs"][3])
    three_handcard = np.array(state["obs"][4])

    legal_actions = np.array(state["legal_actions"])
    cardstr =""
    one_last_action = ""
    two_last_action = ""
    three_last_action=""

    for i in range(15):
        cardnum = handcard[:,i].tolist().index(1)
        one_cardnum = one_handcard[:, i].tolist().index(1)
        two_cardnum = two_handcard[:, i].tolist().index(1)
        three_cardnum = three_handcard[:, i].tolist().index(1)

        if cardnum != 0:
            for mm in range(cardnum):
                cardstr = cardstr + NumtoCard[i]

        if one_cardnum != 0:
            for mm in range(one_cardnum):
                one_last_action = one_last_action + NumtoCard[i]

        if two_cardnum != 0:
            for mm in range(two_cardnum):
                two_last_action = two_last_action + NumtoCard[i]

        if three_cardnum != 0:
            for mm in range(three_cardnum):
                three_last_action = three_last_action + NumtoCard[i]


    return cardstr,one_last_action,two_last_action,three_last_action,legal_actions

def rltorh(card):
    curreny_hand = []
    for i in range(len(card)):
        if card[i] is 'T':
            curreny_hand.append('10')
        elif card[i] is 'B':
            curreny_hand.append('*')
        elif card[i] is 'R':
            curreny_hand.append('$')
        else:
            curreny_hand.append(card[i])
    return curreny_hand
def rhtorl(putcard):
    if len(putcard) == 0:
        action = 'pass'
    else:
        action = ''
        cardindex = []
        for i in range(len(putcard)):
            cardindex.append(cardsIndex[putcard[i]])
        cardindex.sort()

        for index in cardindex :
            action = action + NumtoCard[index]

    return action



def actiontostr(actions):
    print("---")

def strtoaction(actions):
    print("####")

if __name__ == "__main__":
    print(numpytostr(state))






