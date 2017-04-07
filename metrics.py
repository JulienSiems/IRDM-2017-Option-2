"""
Metrics Functions
E.g. AP,MAP,DCG, & NDCG

Coded by Lewis Moffat and Julien Siems 
3/17/2017
"""
import numpy as np
import pdb

def p_n(rel, n):
    """
    This function provides the average precision where relevance is non-binary
    Args:
        rel: relevance score e.g. [0,0,1,0,1,1,0]
        n: what position to go to 
    Returns:
        Precision score at n
    """        
    # filters out zeros
    #rel = np.array(rel)[:n] != 0
    # calculate and return mean
    return np.mean(rel[:n])
    
def AP(rel):
    """
    This function provides the average precision where relevance is non-binary
    Args:
        rel: relevance score e.g. [0,0,1,0,1,1,0]
    Returns:
        Average Precision score
    """
    
    # Convert all values above 1 to 1 so its binary
    rel[rel>1]=1
    # run through each values of k 
    final = [p_n(rel, n + 1) for n in range(len(rel))]
    return np.mean(final)
    
def MAP(rels):
    """
    The mean of average precisions
    Args:
        rel: relevance score e.g. [[0,0,1,0,1,1,0]'[0,1,1,0,0,1,0]]
    Returns:
        Mean of Average Precision scores
    """
    return np.mean([AP(rel) for rel in rels])
    
def DCG(rel,n):
    """
    The discounted cummulative gain at n
    Args:
        rel: relevance score e.g. [[0,0,1,0,1,1,0]'[0,1,1,0,0,1,0]]
        n: value to run up until
    Returns:
        The DCG at n
    """
    # filter zeros
    rel = np.array(rel, dtype='float')[:n]
    return np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))

    

def NDCG(rel,n):
    """
    The normalized discounted cummulative gain at n
    Args:
        rel: relevance score e.g. [[0,0,1,0,1,1,0]'[0,1,1,0,0,1,0]]
        n: value to run up until
    Returns:
        The NDCG at n
    """
    
    maxVal = DCG(sorted(rel, reverse=True), n)
    return DCG(rel, n) / maxVal

    
    
