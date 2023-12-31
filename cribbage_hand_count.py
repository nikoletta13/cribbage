"""
Count points in a hand for cribbage. 
"""
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chisquare, skew


class Card:
    """
    define card from a face and a suit. Input face a number for number cards and JQK for others.
    Face_value: the number on the card for number cards, 11 12 13 for J Q K
    Value: number on the card for number cards, 10 for J Q K
    Suit: the suit of the card
    """
    def __init__(self,face,suit):
        self.face = face
        self.suit = suit
        self.value()
        self.face_value()


    def value(self):
        faces = ['J','Q','K']
        if self.face in faces:
            return 10
        else:
            return  int(self.face)
    def face_value(self):
        faces = ['J','Q','K']
        face_faces = {'J':11,'Q':12,'K':13}
        if self.face not in faces:
            return self.value()
        else:
            return face_faces[self.face]





class Hand:
    """
    A hand is made up of 4 + 1 Cards. Last card is the drawn card.
    """
    def __init__(self,handset):
        self.cards = set()
        self.points = 0
        self.handset = handset
        self.handvalues = [c.value() for c in self.handset]
        self.handfaces = [c.face_value() for c in self.handset]
        self.handsuits = [c.suit for c in self.handset]


    def evaluate_strict_suits(self):
        """
        Get suits of the hand without the top card. 
        """
        return self.handsuits[:-1]

    def evaluate_points(self):
        """
        Collect all the points in the hand. 
        """
        p=0
        # Fifteen
        for i in range(2,6):
            combs_length_i = combinations(self.handvalues,i)
            for comb in combs_length_i:
                if sum(list(comb)) == 15:
                    p+=2

        # Pairs
        for f_c in set(self.handfaces):
            if self.handfaces.count(f_c)==2:
                p+=2
            elif self.handfaces.count(f_c)==3:
                p+=6
            elif self.handfaces.count(f_c)==4:
                p+=12

        # Sequence
        sorted_faces = sorted(self.handfaces)
        max_len = len(max(np.split(sorted_faces, np.where(np.diff(sorted_faces)!=1)[0]+1),key=len))
        if max_len>2:
            p+=max_len
            

        # Flush
        if len(set(self.handsuits)) == 1:
            p+=5
        elif len(set(self.handsuits)) == 2:
            deal_suits = self.evaluate_strict_suits()
            if len(set(deal_suits)) == 1:
                p+=4

        # One for his nob
        top_suit = self.handsuits[-1]
        for i in range(4):
            if self.handsuits[i] == top_suit and self.handfaces[i]==11:
                p+=1

        return p
    



if __name__=='__main__':
        

    SUITS = ['H','C','S','D']


    DECK = [Card(i,s) for i in range(1,11) for s in SUITS]
    DECK += [Card(i,s) for i in ['J','Q','K'] for s in SUITS]

    H4 = combinations(DECK,4)

    def all_hands():
        """
        Use generator to make this faster.
        Can be used to extract list if so desired.
        """
        for h_h in H4:
            top_cards = [t for t in DECK if t not in h_h]
            for t_c in top_cards:
                yield h_h+(t_c,)


    total_points = [Hand(h).evaluate_points() for h in all_hands()]



    def normalise(l,N=1):
        """
        Nrmalise list l to be used for comparison with a sample of size N.
        """
        return [N*x/sum(l) for x in l]
    
    # Plots

    rand_freq = [total_points.count(j)/len(total_points) for j in range(0,30)]
    rand_count_2130 = [total_points.count(j)/len(total_points) for j in range(21,30)]

    df = pd.DataFrame({'dist':rand_count_2130,'score': [j for j in range(21,30)]} )


    sns.set_theme(style="darkgrid")
    sns.catplot(rand_freq, kind='bar',color=sns.color_palette()[2])
    plt.title('Points distribution with random choice')
    plt.xlabel('points')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()



    sns.set_theme(style="darkgrid")
    sns.catplot(data = df,x='score',y='dist', kind='bar',color=sns.color_palette()[2])
    plt.title('Points distribution with random choice for scores between 21 and 29')
    plt.xlabel('points')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()



    with open(r'hand.txt','r') as file:
        pre = file.read().split(',')

    hand = [int(val) for val in pre]

    with open(r'crib.txt','r') as file:
        pre = file.read().split(',')

    crib = [int(val) for val in pre]
    
    hand_count = [hand.count(j)/len(hand) for j in range(0,30)]
    crib_count = [crib.count(j)/len(crib) for j in range(0,30)]

    sns.set_theme(style="darkgrid")
    sns.catplot(data=hand_count, kind='bar',color=sns.color_palette()[0])
    plt.title('Hand points distribution')
    plt.xlabel('points')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()
    
    sns.set_theme(style="darkgrid")
    sns.catplot(data=crib_count, kind='bar',color=sns.color_palette()[1])
    plt.title('Crib points distribution')
    plt.xlabel('points')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()



    conc_scores = [j for j in range(0,30)] + [j for j in range(0,30)] 
    conc_freq = hand_count + crib_count
    origin_labels = ['hand' for j in range(0,30)] + ['crib' for j in range(0,30)] 


    data_df = ({'score':conc_scores, 'origin': origin_labels, 'frequency': conc_freq})


    sns.set_theme(style="darkgrid")
    sns.catplot(data_df, x = 'score', y = 'frequency',hue='origin',legend=True , legend_out=False, kind='bar')
    plt.title('Hand and crib points distribution')
    plt.xlabel('points')
    plt.legend(loc="upper right")
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()


    conc_freq = hand_count + rand_freq
    origin_labels = ['hand' for j in range(0,30)] +['random choice' for j in range(0,30)]


    df_handtotal = ({'score':conc_scores, 'origin': origin_labels, 'frequency': conc_freq})

    sns.set_theme(style="darkgrid")
    sns.catplot(df_handtotal, x = 'score', y = 'frequency',hue='origin', legend=True , legend_out=False, kind='bar', palette=[sns.color_palette()[0],sns.color_palette()[2]])
    plt.title('Hand and random choice points distribution')
    plt.xlabel('points')
    plt.legend(loc="upper right")
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()


    conc_freq = crib_count + rand_freq
    origin_labels = ['crib' for j in range(0,30)] + ['random choice' for j in range(0,30)]


    df_cribtotal = ({'score':conc_scores, 'origin': origin_labels, 'frequency': conc_freq})
   
    sns.set_theme(style="darkgrid")
    sns.catplot(df_cribtotal, x = 'score', y = 'frequency',hue='origin',legend=True , legend_out=False, kind='bar', palette=[sns.color_palette()[1],sns.color_palette()[2]])
    plt.title('Crib and random choice points distribution')
    plt.xlabel('points')
    plt.legend(loc="upper right")
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()

    #------------
    # Statistics
    #------------
    hand_full = [hand.count(j) for j in range(0,30)]
    crib_full = [crib.count(j) for j in range(0,30)]
    # normalise
    rand_full_hand = [x*len(hand) for x in rand_freq]
    rand_full_crib = [x*len(crib) for x in rand_freq]

    hand_mean = np.mean(hand)
    crib_mean = np.mean(crib)
    total_mean = np.mean(total_points)

    hand_var = np.var(hand)
    crib_var = np.var(crib)
    total_var = np.var(total_points)

    df_chi = pd.DataFrame({'random_hand':rand_full_hand,'random_crib':rand_full_crib,'hand':hand_full,'crib':crib_full})

    df_chi = df_chi.drop(df_chi[df_chi['random_hand']==0].index)


    chi_hand = chisquare(f_obs=df_chi['hand'], f_exp=df_chi['random_hand'])#, ddof=len(df_chi)-1)
    chi_crib = chisquare(f_obs=df_chi['crib'], f_exp=df_chi['random_crib'])#, ddof=len(df_chi)-1)

    print(hand_mean,crib_mean,total_mean)
    print(hand_var,crib_var,total_var)
    print(chi_hand)
    print(chi_crib)
    

    # even vs odd
    def even_odd(origin):
        even = [origin[j] for j in range(0,30) if j%2==0]
        odd = [origin[j] for j in range(0,30) if j%2==1]
        p_even = sum(even)
        return even, odd, p_even

    hand_even, hand_odd, p_even_hand = even_odd(hand_count)
    crib_even, crib_odd, p_even_crib = even_odd(crib_count)
    rand_even, rand_odd, p_even_rand = even_odd(rand_freq)

    print(p_even_hand,p_even_crib,p_even_rand)

    # chi sq
    hand_full_even, hand_full_odd, tot_hand_even = even_odd(hand_full) 
    crib_full_even, crib_full_odd, tot_crib_even = even_odd(crib_full) 



    #normalise
    rand_even_hand = normalise(rand_even,tot_hand_even)
    rand_odd_hand = normalise(rand_odd,len(hand)-tot_hand_even)
    rand_even_crib = normalise(rand_even,tot_crib_even)
    rand_odd_crib = normalise(rand_odd,len(crib)-tot_crib_even)


    df_chi_even = pd.DataFrame({'random_hand':rand_even_hand,'random_crib':rand_even_crib,'hand':hand_full_even,'crib':crib_full_even})
    df_chi_odd = pd.DataFrame({'random_hand':rand_odd_hand,'random_crib':rand_odd_crib,'hand':hand_full_odd,'crib':crib_full_odd})

    df_chi_even = df_chi_even.drop(df_chi_even[df_chi_even['random_hand']==0].index)
    df_chi_odd = df_chi_odd.drop(df_chi_odd[df_chi_odd['random_hand']==0].index)


    chi_hand_even = chisquare(f_obs=df_chi_even['hand'], f_exp=df_chi_even['random_hand'])
    chi_crib_even = chisquare(f_obs=df_chi_even['crib'], f_exp=df_chi_even['random_crib'])
    
    chi_hand_odd = chisquare(f_obs=df_chi_odd['hand'], f_exp=df_chi_odd['random_hand'])
    chi_crib_odd = chisquare(f_obs=df_chi_odd['crib'], f_exp=df_chi_odd['random_crib'])

    print(chi_hand_even)
    print(chi_crib_even)
    print(chi_hand_odd)
    print(chi_crib_odd)


    # determine skew
    print(skew(hand))
    print(skew(crib))
    print(skew(total_points))