# -*- coding: utf-8 -*-
''' Implement Doudizhu Dealer class
'''
import functools
import random
from rlcard.utils import init_54_deck
from rlcard.games.doudizhu.utils import cards2str, doudizhu_sort_card
import numpy as np

from rlcard.games.base import Card

class DoudizhuDealer:
    ''' Dealer will shuffle, deal cards, and determine players' roles
    '''
    def __init__(self, np_random):
        '''Give dealer the deck

        Notes:
            1. deck with 54 cards including black joker and red joker
        '''
        self.np_random = np_random
        self.deck = init_54_deck()
        self.dd = []
        xx00 = ['3','3','4', '5', '6', '6', '7', '8', '9', 'T', 'J', 'Q', 'Q', 'K', 'K', 'A', '2','3', '6', '7', '8', '9', '9', 'T', 'T', 'J', 'J', 'Q', 'Q', 'K', 'K', 'A', '2', '', '3', '4', '4', '4', '5', '5', '5', '6', '7', '8', '8', '9', 'T', 'J', 'A', '2', '','7', 'A', '2']
        xx01 = ['S','H','S', 'S', 'S', 'H', 'S', 'S', 'S', 'S', 'S', 'S', 'H', 'S', 'H', 'S', 'S','D', 'D', 'H', 'H', 'H', 'D', 'H', 'D', 'H', 'D', 'D', 'C', 'D', 'C', 'H', 'H', 'BJ', 'C', 'H', 'D', 'C', 'H', 'D', 'C', 'C', 'D', 'D', 'C', 'C', 'C', 'C', 'D', 'D', 'RJ','C', 'C', 'C']

        for i in range(len(xx00)):
            self.dd.append(Card(xx01[i],xx00[i]))

        self.deck.sort(key=functools.cmp_to_key(doudizhu_sort_card))
        self.landlord = None
        self.rn = ''
        self.cards_id = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12, 'B': 13, 'R': 14}

    def shuffle(self):
        ''' Randomly shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, players):
        ''' Deal cards to players
        Args:
            players (list): list of DoudizhuPlayer objects
        '''

        hand_num = (len(self.deck) - 3) // len(players)

        for index, player in enumerate(players):
            current_hand = self.deck[index*hand_num:(index+1)*hand_num]
            current_hand.sort(key=functools.cmp_to_key(doudizhu_sort_card))
            player.set_current_hand(current_hand)
            player.initial_hand = cards2str(player.current_hand)
        """
        for index, player in enumerate(players):
            current_hand = self.dd[index * hand_num:(index + 1) * hand_num]
            current_hand.sort(key=functools.cmp_to_key(doudizhu_sort_card))
            player.set_current_hand(current_hand)
            player.initial_hand = cards2str(player.current_hand)
        """

    def determine_role(self, players):
        ''' Determine landlord and peasants according to players' hand

        Args:
            players (list): list of DoudizhuPlayer objects

        Returns:
            int: landlord's player_id
        '''
        # deal cards
        self.shuffle()
        self.deal_cards(players)
        players[0].role = 'landlord'
        self.landlord = players[0]
        players[1].role = 'peasant'
        players[2].role = 'peasant'
        #players[0].role = 'peasant'
        #self.landlord = players[0]

        ## determine 'landlord'
        #max_score = get_landlord_score(
        #    cards2str(self.landlord.current_hand))
        #for player in players[1:]:
        #    player.role = 'peasant'
        #    score = get_landlord_score(
        #        cards2str(player.current_hand))
        #    if score > max_score:
        #        max_score = score
        #        self.landlord = player
        #self.landlord.role = 'landlord'

        # give the 'landlord' the  three cards

        self.landlord.current_hand.extend(self.deck[-3:])

        """
        self.landlord.current_hand.extend(self.dd[-3:])
        """

        self.landlord.current_hand.sort(key=functools.cmp_to_key(doudizhu_sort_card))
        self.landlord.initial_hand = cards2str(self.landlord.current_hand)
        return self.landlord.player_id

    def my_str_to_martix(self,handcards):
        mm = np.zeros([1, 4, 15], dtype=np.int)
        for item in handcards:
            c_id = self.cards_id[item]
            for i in range(4):
                if mm[0][i][c_id] == 0:
                    mm[0][i][c_id] = 1
                    break
        return mm


    def m_determine_role(self, players):
        ''' Determine landlord and peasants according to players' hand

                Args:
                    players (list): list of DoudizhuPlayer objects

                Returns:
                    int: landlord's player_id
                '''

        self.shuffle()
        self.deal_cards(players)
        hand = players[0].initial_hand
        hands_maritx = self.my_str_to_martix(hand)

        role = self.rn.choose_role(np.expand_dims(hands_maritx, 0))

        p_id = role

        if p_id == 0:
            players[0].role = 'landlord'
            self.landlord = players[0]
            players[1].role = 'peasant'
            players[2].role = 'peasant'
        elif p_id == 1:
            players[2].role = 'landlord'
            self.landlord = players[2]
            players[0].role = 'peasant'
            players[1].role = 'peasant'
        else:
            players[1].role = 'landlord'
            self.landlord = players[1]
            players[2].role = 'peasant'
            players[0].role = 'peasant'



        self.landlord.current_hand.extend(self.deck[-3:])
        self.landlord.current_hand.sort(key=functools.cmp_to_key(doudizhu_sort_card))
        self.landlord.initial_hand = cards2str(self.landlord.current_hand)

        #print("role==",role,"========",self.landlord.player_id)

        return self.landlord.player_id
