import numpy as np
import pandas as pd
from numpy import log2 as log
smallvalue = np.finfo(float).eps



X = pd.read_csv('data1_19.csv')


def entropy(inp):
    Class = inp.keys()[-1]   
    entropy = 0
    values = inp[Class].unique()
    for value in values:
        fraction = inp[Class].value_counts()[value]
        fraction=fraction/len(inp[Class])
        temp=-fraction*np.log2(fraction)
        entropy=entropy+temp
        #entropy += -fraction*np.log2(fraction)
    return entropy
  
# Function to calculate entropy of attribute
def entropy_att(inp,attribute):
  Class = inp.keys()[-1] 
  target_variables = inp[Class].unique()  
  variables = inp[attribute].unique()    
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          fraction = len(inp[attribute][inp[attribute]==variable][inp[Class] ==target_variable])/(len(inp[attribute][inp[attribute]==variable])+smallvalue)
          fraction=-fraction*log(fraction+smallvalue)
          entropy=entropy+fraction
      fraction2 = len(inp[attribute][inp[attribute]==variable])/len(inp)
      fraction2=fraction2*entropy
      entropy2 += -fraction2
  if entropy2>0:
  		return entropy2
  else:
  		return entropy2*-1

# Find the decider node
def decider_node(inp):
   # Entropy_att = []
    IG=[]
    
    for key in inp.keys()[:-1]:
        IG.append(entropy(inp)-entropy_att(inp,key))
    max_val=-1
    itr=0
    for val in IG:
    		if max_val<val:
    			max_val=val
    			ind=itr
    		itr=itr+1
    return inp.keys()[:-1][ind]
  
#def get_subtable(inp, node,value):
 # return inp[inp[node] == value].reset_index(drop=True)

# Build Information gain Decision Tree
def buildTree(inp,count,maxlevels,tree=None): 
  #  Class = inp.keys()[-1]  
    node = decider_node(inp)
    attValue = np.unique(inp[node])    
    if tree is None:                    
        tree={}
        tree[node] = {}
    for value in attValue: 
    
        subtable = inp[inp[node] == value].reset_index(drop=True)
        clValue,counts = np.unique(subtable['survived'],return_counts=True)                         
        if len(counts)==1:
        		tree[node][value] = clValue[0]
        		continue
        if count==maxlevels:
            tree[node][value] = clValue[0]    
            continue                                                
        else:        
            tree[node][value] = buildTree(subtable,count+1,maxlevels)
                   
    return tree

def print_tree(dic,level):
    if type(dic)==dict:
    	print()
    else:
        print(":"+dic)
        return
    for key in dic:
        val = dic[key]
        if type(val)!=dict:
        		continue
        else:
            for k in val:
            	
                for i in range(level):
                    print("\t",end="")
                print("|"+key+"="+str(k),end="")
                print_tree(val[k],level+1)


tree = buildTree(X ,0,2)
print_tree(tree,0)
