import spacy
from spacy import load
from typing import NamedTuple
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus import stopwords 
import re
import pandas as pd
nlp = load("en_core_web_sm")
import warnings
#Import PhraseMatcher and create a matcher object:
from spacy.matcher import PhraseMatcher
#Here we create Spans from each match, and create named entities from them:
from spacy.tokens import Span
warnings.filterwarnings('ignore')

#python -m spacy download en_core_web_lg



#load words form csv file
words_list=[]
video_list=[]
df = pd.read_csv('finalcsvfileforserver.csv')
words_list = words_list + df['Video Title'].tolist()
video_list = video_list + df['Video Key'].tolist()





class ISLToken(NamedTuple):
    """Class to hold ISL token with relevant syntactic information"""
    
    text: str
    orig_id: int
    dep: str
    head: int
    tag: str
    ent_type: str
    children: list

        
def filter_spans(spans):
  """Filter a sequence of spans so they don't contain overlaps"""
  get_sort_key = lambda span: (span.end - span.start, -span.start)
  sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
  result = []
  seen_tokens = set()
  for span in sorted_spans:
    # Check for end - 1 here because boundaries are inclusive
    if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
      result.append(span)
      seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def token_chunker(doc):
  """Merge entities and noun chunks into one token""" 

  spans = list(doc.ents) + list(doc.noun_chunks)
  spans = filter_spans(spans)
  if spans != None:
    with doc.retokenize() as retokenizer:
      for span in spans:
        retokenizer.merge(span)


def cc_chunker(doc):
  """
    Merge cc (only 'and' for now) conjuncions for like elements. To be run
    after token_chunker.
    returns -1 if and is chunked, or the token index if sentence is to be split
  """    
    
  for token in doc:
    i = token.i
    if (token.dep_ == "cc"): # and (token.text.lower() == "and"):
      if i == 0:
        return 0
            
      # if head is attached to the 'and', merge the phrase
      if (token.head.i == i-1 and token.head.tag_[:2] != 'VB'):
        # debug:
        # print("merging and chunk:",token.head.left_edge.text, token.head.right_edge.text)
        and_span = doc[token.head.left_edge.i : token.head.right_edge.i + 1]
        with doc.retokenize() as retokenizer:
          retokenizer.merge(and_span)
                   
        # no need to split
        return -1
            
      # else return 'and' index to split 
      return i
    
  # default to no split
  return -1


def ques_conj(doc):
    
  for token in doc:
    i = token.i
    if (token.dep_ == "mark"): 
      if i == 0:
        return 0
            
      # if head is attached to the 'and', merge the phrase
      '''if (token.i != 0):
        #start of the sentence
        return -1            
      # else return 'and' index to split''' 
      return i
    
  # default to no split
  return -1


#It'll replace all numbers like 2020 with two zero two zero
def replace_num(text):
  'Number replacement'
  text = re.sub('0',' zero',text)
  text = re.sub('1',' one',text)
  text = re.sub('2',' two',text)
  text = re.sub('3',' three',text)
  text = re.sub('4',' four',text)
  text = re.sub('5',' five',text)
  text = re.sub('6',' six',text)
  text = re.sub('7',' seven',text)
  text = re.sub('8',' eight',text)
  text = re.sub('9',' nine',text)
  return text


#Adding indian names
df = pd.read_csv('Indian-Male-Names.csv')
names = df['name'].tolist()
df = pd.read_csv('Indian-Female-Names.csv')
names = names +  df['name'].tolist()
for i in range(len(names)):
  try:
    if " " in names[i]:
      names[i] = names[i].split(" ")[0]
  except:
    continue
names = list(set([name for name in names if str(name)!= "nan" ]))
names2 = [name[0].upper() + name[1:] for name in names]
names = names + names2



def add_name(doc, text,names):
  matcher = PhraseMatcher(nlp.vocab)
  #Create the desired phrase patterns:
  terms = {token.text for token in doc if token.tag_ == "NNP"}
  names = {name for name in names}
  names = list(names.intersection(terms))
  phrase_list = names #['Ram', 'Laxman', 'Ajay', 'Sparsh'] #names
  phrase_patterns = [nlp(text) for text in phrase_list]
  #Apply the patterns to our matcher object:
  matcher.add('People, including fictional', None, *phrase_patterns)
  #Apply the matcher to our Doc object:
  matches = matcher(doc)
  return matches
  #See what matches occur: matches #output: [(2689272359382549672, 7, 9), (2689272359382549672, 14, 16)]


def labeller(doc):
  PERSON = doc.vocab.strings[u'PERSON'] 
  matches = add_name(doc,text,names)
  new_ents = [Span(doc, match[1],match[2],label=PERSON) for match in matches if any(t.ent_type for t in doc[match[1]:match[2]])== False] #match[1] contains the start index of the the token and match[2] the stop index (exclusive) of the token in the 
  #if not new_ents in doc. doc.ents:
  doc. doc.ents = list(doc.ents) + new_ents 
  #show_ents(doc)


def eng_isl_translate(doc): #, box_list):
  """Function to translate English to ISL gloss"""  
  
  
  # init lists
  dep_list = []
  type_list = []
  tag_list = []   
  ISLTokens = []
  done_list = []

  #Function to add new labels

  
  labeller(doc)
  
  
  # debug:
  # for token in doc:
  #     print(token.text, token.dep_, token.head.text, token.tag_, token.ent_type_,
  #           [child for child in token.children])

  # chunk noun phrases and entities
  
  token_chunker(doc)



  dep_list = [token.dep_ for token in doc]
  

  flag = 0
  k=0
  doc2_comma = None
  if 'mark' not in dep_list[0]:
    for token in doc:
      k+=1
      if (token.text == "," or token.text == ";") and (token.head.dep_ == "ROOT" or token.head.dep_ == "VBD"):
        #print(token.text,token.i, doc[k].dep_,k)
        if flag ==0:
          punct_i = token.i
          doc2_comma_root_i = doc[punct_i + 1 : ].root.i - punct_i - 1
          doc2_comma = doc[punct_i +1: ].as_doc()
          # set 'ROOT'
          doc2_comma[doc2_comma_root_i].dep_ = "ROOT"

          
          doc_root_i = doc[ : punct_i].root.i
          doc = doc[ : punct_i].as_doc()
          doc[doc_root_i].dep_ = "ROOT"

          ISLTokens = eng_isl_translate(doc)
          ISLTokens2 = eng_isl_translate(doc2_comma)
          ISLTokens.append(nlp("-pause-"))
          ISLTokens.extend(ISLTokens2)
          return ISLTokens
        break



  
  # used to split cc'd clauses
  doc2 = None
  and_tkn = None
  
  # check for cc and process it
  for token in doc:
    if "CC" == token.tag_ :# and doc[token.i-1].text !=',':
      and_i = cc_chunker(doc)         
      # if sentence needs to be split to clauses
      if and_i > -1:
          
        # get 'root' of second piece
        if doc[and_i + 1].text == ",":
          doc2_root_i = doc[and_i + 2 : ].root.i - and_i - 2
          
          doc2 = doc[and_i + 2 : ].as_doc()
        else:
          doc2_root_i = doc[and_i + 1 : ].root.i - and_i - 1
          
          doc2 = doc[and_i + 1 : ].as_doc()
        
        # set 'ROOT'
        doc2[doc2_root_i].dep_ = "ROOT"
        
        # get the cc token
        and_tkn = doc[and_i]
        
        # truncate original doc
        # we need to account for the case where a seentence starts with 'and'
        if and_i == 0:
          
          ISLTokens2 = eng_isl_translate(doc2)
          ISLTokens.append(and_tkn)
          ISLTokens.extend(ISLTokens2)
          return ISLTokens
            
        # the regular case:
        doc = doc[0 : and_i].as_doc()
      #CHECK MORE EXAMPLES
      #For cases when comma followed by CC
      if doc[0].tag_ == "CC":
        new_r_i = doc[1:].root.i - 1
        doc = doc[1 : ].as_doc()
        doc[new_r_i].dep_ = "ROOT" 
        
          
      # break at first 'and', as the process is recursive
      break




  
  #When first word is since/because

  doc2_ques = None
  ques_tkn = None
  flag = 0
  for token in doc:
    if "mark" == token.dep_ :
      mark_i = ques_conj(doc)     
      #FIX THIS ISSUE, ROOT not identified.
      
      # if sentence needs to be split to clauses
      #If since/because at the start
      if mark_i == 0:
        flag = 1
        for tkn in doc:
          if tkn.dep_ == "punct" and (tkn.head.dep_ == "ROOT" or tkn.head.tag_[:2] == "VB"):
            punct_i = tkn.i #dep_tag[tkn.i]
            break
        # get 'root' of second piece
        doc2_root_i = doc[punct_i + 1 : ].root.i - punct_i - 1
        
        doc2_ques = doc[punct_i +1 : ].as_doc()
        
        #print(doc)
        
        # set 'ROOT'
        doc2_ques[doc2_root_i].dep_ = "ROOT"
        
        # get the cc token
        #coma = nlp('')
        comma_tkn =  doc[punct_i] #comma

        doc_root_i = doc[1 : punct_i].root.i
        doc = doc[1 : punct_i].as_doc()
        doc[doc_root_i-1].dep_ = "ROOT"
        

        # truncate original doc
        # we need to account for the case where a seentence starts with 'and'
        if mark_i == 0:
          # debug:
          # print("Initial and")
          # print(and_tkn.text, and_tkn.dep_, and_tkn.head.text,
          #       and_tkn.tag_, and_tkn.ent_type_,
          #       [child for child in and_tkn.children])
          
          ISLTokens = eng_isl_translate(doc2_ques)
          ISLTokens2 = eng_isl_translate(doc)
          ISLTokens.append(nlp("why"))
          ISLTokens.append(nlp("-pause-"))
          ISLTokens.extend(ISLTokens2)
          return ISLTokens
        # the regular case:
        #doc = doc[1 : punct_i].as_doc()
        
        
      # break at first 'mark', as the process is recursive (for middle so/because)
      elif mark_i > 0:
        
        mark_ii = -1
        for tkn in doc:
          mark_ii +=1
          if tkn.dep_ == "mark":
            break

 
 

        # get 'root' of second piece
        if doc[mark_ii + 1].dep_ == 'punct':
          doc2_root_i = doc[mark_ii + 2 : ].root.i - mark_ii - 2
          doc2_ques = doc[mark_ii +2: ].as_doc()
        else:
          doc2_root_i = doc[mark_ii + 1 : ].root.i - mark_ii - 1
          doc2_ques = doc[mark_ii +1: ].as_doc()
        # set 'ROOT'
        doc2_ques[doc2_root_i].dep_ = "ROOT"
        
        # get the cc token
        if tkn.text in ['that']:
          ques_tkn = nlp("what")
        else:
          ques_tkn =  nlp("why") #doc[mark_ii]

        # truncate original doc
        # we need to account for the case where a seentence starts with 'and'
        if mark_i <0:
          # debug:
          # print("Initial and")
          # print(and_tkn.text, and_tkn.dep_, and_tkn.head.text,
          #       and_tkn.tag_, and_tkn.ent_type_,
          #       [child for child in and_tkn.children])
          
          #ISLtokens = eng_isl_translate(doc[0 : mark_ii].as_doc())
          ISLTokens2 = eng_isl_translate(doc2_ques)
          ISLTokens.append(ques_tkn)
          ISLTokens.extend(ISLTokens2)
          return ISLTokens
        #CHECK ERROR
        # the regular case:
        #####0 ka 1
        doc = doc[0 : mark_ii].as_doc()
        
      break 
 
  
  #for simple comma
  '''doc2_comma = None
  for token in doc:
    if (token.text == "," or token.text == ";") and (token.head.dep_ == "ROOT" or token.head.dep_ == "VBD"):
      if flag ==0:
        punct_i = token.i
        doc2_comma_root_i = doc[punct_i + 1 : ].root.i - punct_i - 1
        doc2_comma = doc[punct_i +1: ].as_doc()
        # set 'ROOT'
        doc2_comma[doc2_comma_root_i].dep_ = "ROOT"
        
        doc_root_i = doc[ : punct_i].root.i
        doc = doc[ : punct_i].as_doc()
        doc[doc_root_i].dep_ = "ROOT"
        ISLTokens = eng_isl_translate(doc)
        ISLTokens2 = eng_isl_translate(doc2_comma)
        ISLTokens.append(nlp("-pause-"))
        ISLTokens.extend(ISLTokens2)
        return ISLTokens
      break'''




  


  

  #Remove full stop for later
  if doc[-1].text == ".":
    doc = doc[:-1].as_doc()
  
    
  dep_list = []
  for token in doc:
    dep_list.append(token.dep_)
    tag_list.append(token.tag_)
    type_list.append(token.ent_type_)
    
      
  # time related words are first in ISL sentences


  #Removing articles and Dropwords
  for token in doc:
    if token.text.lower() in ['a','an','the'] and token.tag_ == "DT":
      done_list.append(token.i)
    elif token.text in ['of']:
      done_list.append(token.i)



  #Added "TIME" entity type

  while "TIME" in type_list:
    time_i = type_list.index("TIME")
    type_list[time_i]=""
    if not time_i in done_list:
      done_list.append(time_i)
      tkn = doc[time_i]
      ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                tkn.ent_type_, [child for child in tkn.children]))
      if doc[time_i].dep_ == "pobj":
        time_ii = doc[time_i].head.i
        if time_ii not in done_list:
          tkn = doc[time_ii]
          done_list.append(time_ii)
          ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i,
                                    tkn.tag_, tkn.ent_type_,
                                    [child for child in tkn.children]))




  if "DATE" in type_list:
    date_i = type_list.index("DATE")
    tkn = doc[date_i]
    if any("WRB" in tknchild.tag_ for tknchild in tkn.children) == False:
      if date_i not in done_list:
        done_list.append(date_i)
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                  tkn.ent_type_, [child for child in tkn.children]))
      if doc[date_i].dep_ == "pobj":
        date_ii = doc[date_i].head.i
        if date_ii not in done_list:
          tkn = doc[date_ii]
          done_list.append(date_ii)
          ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i,
                                    tkn.tag_, tkn.ent_type_,
                                    [child for child in tkn.children]))
      




  # place related words are next in ISL reference

  if "GPE" in type_list:
    place_i = type_list.index("GPE")
    done_list.append(place_i)
    tkn = doc[place_i]
    ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                              tkn.ent_type_, [child for child in tkn.children]))
    
    

  # object attribute in their children to add as adjective
  #obj_adj_i = dep_list.index('amod')
  if "EX" in tag_list:
    ex_i = tag_list.index("EX")
    if not ex_i in done_list:
      done_list.append(ex_i)
      tkn = doc[ex_i]
      ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                              tkn.ent_type_, [child for child in tkn.children]))


  # place related but it's an object following a preposition (eg under the tree, besides the building, etc) 
  pobj_flag_amod = 0
  pobj_flag_poss = 0
  if "pobj" in dep_list:
    pobj_i = dep_list.index("pobj")
    tkn = doc[pobj_i]
    if tkn.head.dep_ == "prep":
      #print("pobj_prep")
      if not tkn.head.i in done_list:
        pobj_flag_prep_i = dep_list.index("prep")
        done_list.append(pobj_flag_prep_i)
        #if tkn.head.text.lower() not in ['to','on','for','along','than']:
        # ISLTokens.append(ISLToken(tkn.head.lemma_, tkn.head.i, tkn.head.dep_, tkn.head.head.i, tkn.head.tag_,
        #                        tkn.head.ent_type_, [child for child in tkn.head.children]))
      if not tkn.i in done_list:
        for pobj_child in tkn.children:

          if not pobj_child.i in done_list:
            if pobj_child.dep_ == "poss":
              pobj_flag_poss_i = dep_list.index("poss")
              #pobj_flag_poss = 1
              done_list.append(pobj_flag_poss_i)
              if pobj_child.lemma_ == '-PRON-':
                ISLTokens.append(ISLToken(pobj_child.text, pobj_child.i, pobj_child.dep_, pobj_child.head.i, pobj_child.tag_,
                                    pobj_child.ent_type_, [child for child in pobj_child.children]))
              else:
                ISLTokens.append(ISLToken(pobj_child.lemma_, pobj_child.i, pobj_child.dep_, pobj_child.head.i, pobj_child.tag_,
                                    pobj_child.ent_type_, [child for child in pobj_child.children]))
                
            if pobj_child.dep_ == "amod":
              #pobj_flag_poss = 1
              #if pobj_flag_amod == 1: 
              pobj_flag_amod_i = dep_list.index("amod")               
              done_list.append(pobj_flag_amod_i)
              ISLTokens.append(ISLToken(pobj_child.text, pobj_child.i, pobj_child.dep_, pobj_child.head.i, pobj_child.tag_,
                                  pobj_child.ent_type_, [child for child in pobj_child.children]))


        done_list.append(pobj_i)
        if tkn.lemma_ == "-PRON-":
          ISLTokens.append(ISLToken(tkn.text, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                            tkn.ent_type_, [child for child in tkn.children]))
        else:
          ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                            tkn.ent_type_, [child for child in tkn.children]))  


        
          

    
          
          
        


  # Checks roots, compares, finds passive person, active person, so passive, active, root(verb)

  try:
    root_i = dep_list.index("ROOT")
  except:
    try:
      root_i = doc.root.i
      doc[root_i].dep = "ROOT"
    except:
      root_i = 0
  root_children = [child for child in doc[root_i].children]
  #root_children = [child for child in doc[root_i].children]  

  


  '''# subjects come next
  subj_type = 0
  if "nsubj" in dep_list or "nsubjpass" in dep_list:
    try:
      nsubj_i = dep_list.index("nsubj")
      subj_type = 0
    except:
      nsubj_i = dep_list.index("nsubjpass")
      subj_type = 1
    tkn = doc[nsubj_i]
    
    
    
    if not tkn.tag_[0] == 'W' and not tkn.i in done_list:
      for nsubj_child in tkn.children:
        if any("W" in str(nsubj_gchild.tag_) for nsubj_gchild in list(nsubj_child.children)) == False:
          if nsubj_child.dep_ == "poss":
            if not nsubj_child.i in done_list:
              done_list.append(nsubj_child.i)
              if nsubj_child.lemma_ == '-PRON-':
                ISLTokens.append(ISLToken(nsubj_child.text, nsubj_child.i, nsubj_child.dep_, nsubj_child.head.i, nsubj_child.tag_,
                                      nsubj_child.ent_type_, [child for child in nsubj_child.children])) 
              else:
                ISLTokens.append(ISLToken(nsubj_child.lemma_, nsubj_child.i, nsubj_child.dep_, nsubj_child.head.i, nsubj_child.tag_,
                                    nsubj_child.ent_type_, [child for child in nsubj_child.children]))     
          if nsubj_child.dep_ == "amod" or nsubj_child.dep_ == "compound":
            if not nsubj_child.i in done_list:
              done_list.append(nsubj_child.i)
              ISLTokens.append(ISLToken(nsubj_child.lemma_, nsubj_child.i, nsubj_child.dep_, nsubj_child.head.i, nsubj_child.tag_,
                                  nsubj_child.ent_type_, [child for child in nsubj_child.children]))
            
        
      done_list.append(nsubj_i)
      if tkn.lemma_ == '-PRON-' or tkn.tag_=="PRP":
        if tkn.lemma_.lower() == 'i':
          ISLTokens.append(ISLToken("me", tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                tkn.ent_type_, [child for child in tkn.children]))
        else:
          ISLTokens.append(ISLToken(tkn.text, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                tkn.ent_type_, [child for child in tkn.children]))
      else:
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                              tkn.ent_type_, [child for child in tkn.children]))
  
    if subj_type == 0:
      dep_list.index('nsubj') == ''
    else:
      dep_list.index('nsubjpass') == ''
  '''
    
  for tkn in root_children:
    if tkn.dep_ == "attr":
        if not tkn.i in done_list:
          for attr_children in tkn.children:
            if not attr_children.i in done_list:
              done_list.append(attr_children.i)
              if attr_children.lemma_ == '-PRON-':
                ISLTokens.append(ISLToken(attr_children.text, attr_children.i, attr_children.dep_, attr_children.head.i, attr_children.tag_,
                                    attr_children.ent_type_, [child for child in attr_children.children])) 
              else:
                ISLTokens.append(ISLToken(attr_children.lemma_, attr_children.i, attr_children.dep_, attr_children.head.i, attr_children.tag_,
                                    attr_children.ent_type_, [child for child in attr_children.children]))
          done_list.append(tkn.i)
          ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                            tkn.ent_type_, [child for child in tkn.children]))


  if "dobj" in dep_list:
    dobj_i = dep_list.index("dobj")
    #dobj_i = root_children[dobj_i_1].i
    tkn = doc[dobj_i]
    if not dobj_i in done_list:
      #Possessive prep for obj
      for dobjchild in tkn.children:
        if dobjchild.dep_ in ["poss","amod", "advcl","det"] and dobjchild.text.lower() not in ['a','an','the']:
          if not dobjchild.i in done_list:
            dobjchild_i = dep_list.index(dobjchild.dep_)
            done_list.append(dobjchild_i)
            ISLTokens.append(ISLToken(dobjchild.text, dobjchild.i, dobjchild.dep_, dobjchild.head.i, dobjchild.tag_,
                                dobjchild.ent_type_, [child for child in dobjchild.children]))

      
      done_list.append(dobj_i)
      if tkn.lemma_ == '-PRON-':
        ISLTokens.append(ISLToken(tkn.text, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                  tkn.ent_type_, [child for child in tkn.children]))
      elif tkn.lemma_.lower() == 'i':
        ISLTokens.append(ISLToken('me', tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                  tkn.ent_type_, [child for child in tkn.children]))                            
      else:
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                  tkn.ent_type_, [child for child in tkn.children]))
        
      if tkn.head.dep_ == "advcl":
        if not tkn.head.i in done_list:
          dobj_verb_i = dep_list.index("advcl") 
          done_list.append(dobj_verb_i)
          ISLTokens.append(ISLToken(tkn.head.text, tkn.head.i, tkn.head.dep_, tkn.head.head.i, tkn.head.tag_,
                          tkn.head.ent_type_, [child for child in tkn.head.children]))








  # checks to see if words are in the following categories and appends them
  if not {"xcomp", "ccomp", "prep","acomp"}.isdisjoint([child.dep_ for child in doc[root_i].children]):
    for child in doc[root_i].children:
      if any("W" in str(gchild.tag_) for gchild in list(child.children)) == False:
        if child.dep_ in ("xcomp", "ccomp", "prep","acomp") :#, "advcl"
          subtree_span = doc[child.left_edge.i : child.right_edge.i + 1]
          for tkn in subtree_span:
            if tkn.tag_[0] != 'W' and tkn.text not in ['to','be']:
              if not tkn.i in done_list:
                if tkn.tag_ == "PRP":
                  ISLTokens.append(ISLToken(tkn.text, tkn.i, tkn.dep_, 
                                            tkn.head.i, tkn.tag_, tkn.ent_type_, 
                                            [child for child in tkn.children]))
                else:
                  ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, 
                                            tkn.head.i, tkn.tag_, tkn.ent_type_, 
                                            [child for child in tkn.children]))
          
                done_list.append(tkn.i)

  '''
  # verb obects usually come last
  #if "dobj" in [child.dep_ for child in doc[root_i].children]:
  #  dobj_i_1 = [child.dep_ for child in doc[root_i].children].index("dobj")
  if "dobj" in dep_list:
    dobj_i = dep_list.index("dobj")
    #dobj_i = root_children[dobj_i_1].i
    tkn = doc[dobj_i]
    if not dobj_i in done_list:
      #Possessive prep for obj
      for dobjchild in tkn.children:
        if dobjchild.dep_ in ["poss","amod", "advcl","det"] and dobjchild.text.lower() not in ['a','an','the']:
          if not dobjchild.i in done_list:
            dobjchild_i = dep_list.index(dobjchild.dep_)
            done_list.append(dobjchild_i)
            ISLTokens.append(ISLToken(dobjchild.text, dobjchild.i, dobjchild.dep_, dobjchild.head.i, dobjchild.tag_,
                                dobjchild.ent_type_, [child for child in dobjchild.children]))
      
      done_list.append(dobj_i)
      if tkn.lemma_ == '-PRON-':
        ISLTokens.append(ISLToken(tkn.text, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                  tkn.ent_type_, [child for child in tkn.children]))
      elif tkn.lemma_.lower() == 'i':
        ISLTokens.append(ISLToken('me', tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                  tkn.ent_type_, [child for child in tkn.children]))                            
      else:
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                  tkn.ent_type_, [child for child in tkn.children]))
        
      if tkn.head.dep_ == "advcl":
        if not tkn.head.i in done_list:
          dobj_verb_i = dep_list.index("advcl") 
          done_list.append(dobj_verb_i)
          ISLTokens.append(ISLToken(tkn.head.text, tkn.head.i, tkn.head.dep_, tkn.head.head.i, tkn.head.tag_,
                          tkn.head.ent_type_, [child for child in tkn.head.children]))'''
  

  #to check current order
  #print([doc[i] for i in done_list])


  #Subject comes next
  subj_type = 0
  if "nsubj" in dep_list or "nsubjpass" in dep_list:
    try:
      nsubj_i = dep_list.index("nsubj")
      subj_type = 0
    except:
      nsubj_i = dep_list.index("nsubjpass")
      subj_type = 1
    tkn = doc[nsubj_i]
    
    
    
    if not tkn.tag_[0] == 'W' and not tkn.i in done_list:
      for nsubj_child in tkn.children:
        if any("W" in str(nsubj_gchild.tag_) for nsubj_gchild in list(nsubj_child.children)) == False:
          if nsubj_child.dep_ == "poss":
            if not nsubj_child.i in done_list:
              done_list.append(nsubj_child.i)
              if nsubj_child.lemma_ == '-PRON-':
                ISLTokens.append(ISLToken(nsubj_child.text, nsubj_child.i, nsubj_child.dep_, nsubj_child.head.i, nsubj_child.tag_,
                                      nsubj_child.ent_type_, [child for child in nsubj_child.children])) 
              else:
                ISLTokens.append(ISLToken(nsubj_child.lemma_, nsubj_child.i, nsubj_child.dep_, nsubj_child.head.i, nsubj_child.tag_,
                                    nsubj_child.ent_type_, [child for child in nsubj_child.children]))     


          if nsubj_child.dep_ == "amod" or nsubj_child.dep_ == "compound":
            if not nsubj_child.i in done_list:
              done_list.append(nsubj_child.i)
              ISLTokens.append(ISLToken(nsubj_child.lemma_, nsubj_child.i, nsubj_child.dep_, nsubj_child.head.i, nsubj_child.tag_,
                                  nsubj_child.ent_type_, [child for child in nsubj_child.children]))
            
        

      done_list.append(nsubj_i)
      if tkn.lemma_ == '-PRON-' or tkn.tag_=="PRP":
        if tkn.lemma_.lower() == 'i':
          ISLTokens.append(ISLToken("me", tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                tkn.ent_type_, [child for child in tkn.children]))
        else:
          ISLTokens.append(ISLToken(tkn.text, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                                tkn.ent_type_, [child for child in tkn.children]))
      else:
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                              tkn.ent_type_, [child for child in tkn.children]))
  
    if subj_type == 0:
      dep_list.index('nsubj') == ''
    else:
      dep_list.index('nsubjpass') == ''












  #Adding ROOT 
  tkn = doc[root_i]
  if not tkn.lemma_ in ['be','for']: # and root_i!=0:
    if tkn.dep_ != 'aux' and root_i!=0:
      done_list.append(root_i)
      ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                              tkn.ent_type_, [child for child in tkn.children]))
  elif tkn.lemma == 'do' and tkn.dep_[:1] ==  "V":
    if tkn.dep_ != 'aux' and root_i!=0:
      done_list.append(root_i)
      ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                              tkn.ent_type_, [child for child in tkn.children]))
  else:
    pass



  #Adverb appended after verb
  for rootchild in tkn.children:
    if rootchild.dep_ == "advmod" and rootchild.tag_ == "RB":
      if any("W" in str(root_gchild.tag_) for root_gchild in list(rootchild.children)) == False:
        if not rootchild.i in done_list:
          done_list.append(rootchild.i)
          ISLTokens.append(ISLToken(rootchild.lemma_, rootchild.i, rootchild.dep_, rootchild.head.i, rootchild.tag_,
                              rootchild.ent_type_, [child for child in rootchild.children]))

  isl_root_i = len(ISLTokens) - 1
  
  
  # auxiliaries like must, can etc. come after the object
  if "aux" in [child.dep_ for child in doc[root_i].children]:
    aux_i_1 = [child.dep_ for child in doc[root_i].children].index("aux")
    aux_i = root_children[aux_i_1].i
    if root_children[aux_i_1].lemma_.lower() not in ['be', 'do','to','should','will','can','may','could','would']:
      tkn = doc[aux_i]
      done_list.append(aux_i)
      ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                              tkn.ent_type_, [child for child in tkn.children]))
    


  #TRYING FOR VERBS AND ADVERBS




  # negatives come last in non-questions
  if "neg" in dep_list:
    neg_i = dep_list.index("neg")
    tkn = doc[neg_i]
    done_list.append(neg_i)
    ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                              tkn.ent_type_, [child for child in tkn.children]))
    

  

  # question markers come dead last
  if tag_list[0][0] == 'W':
    token = doc[0]
    #For How questions ending with What
    if token.head.dep_ == "amod" or token.head.dep_ == "advmod" or token.head.dep_ == "acomp":
      if not token.head.i in done_list:
        what = "what"
        done_list.append(token.head.i)
        ISLTokens.append(ISLToken(token.head.lemma_, token.head.i, token.head.dep_, token.head.head.i, token.head.tag_,
                                 token.head.ent_type_, [child for child in token.head.children]))
        done_list.append(token.i)
        ISLTokens.append(ISLToken(what, token.i, token.dep_, token.i, token.tag_,
                                    token.ent_type_, [child for child in token.children]))
    else:
      if not 0 in done_list:
        done_list.append(token.i)
        ISLTokens.append(ISLToken(token.lemma_, token.i, token.dep_, token.i, token.tag_,
                                    token.ent_type_, [child for child in token.children]))
  #print([doc[i] for i in done_list])
    
  #How questions      
  if doc[0].lemma_ in ['be','do'] or doc[0].tag_ == "MD":
    token = doc[0]
    possible = "-possible-"
    done_list.append(token.i)
    ISLTokens.append(ISLToken(possible, token.i, token.dep_, token.i, token.tag_,
                                  token.ent_type_, [child for child in token.children]))




  j = isl_root_i
  # insert children of ROOT next to it
  for tkn in root_children:
    if not tkn.i in done_list:
      if tkn.lemma_ not in ['be','to']:
        if not tkn.dep_ in ["aux", "punct", "neg","auxpass"]:
          done_list.append(tkn.i)
          ISLTokens.insert(j, ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i,
                                        tkn.tag_, tkn.ent_type_,
                                        [child for child in tkn.children]))
          j += 1



  # insert the remainders after
  for tkn in doc:
    if not tkn.i in done_list:
      if tkn.lemma_ not in ['be','to']:
        if not tkn.dep_ in ["aux", "punct", "neg","poss"]:
          done_list.append(tkn.i)
          ISLTokens.insert(j, ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i,
                                        tkn.tag_, tkn.ent_type_,
                                        [child for child in tkn.children]))
          j += 1
    


    


  #Add tense
  
  '''for token in doc:
    if token.tag_[:2] == 'VB':
      tense_place = token.i + 1
      if token.lemma_ != "be":
        if token.morph.get('Tense') == ['Pres']:
          ISLTokens.insert(tense_place, nlp('(present)'))
        if token.morph.get('Tense') == ['Past']:
          ISLTokens.insert(tense_place, nlp('(past)'))
      else:
        if token.children != []:
          if any('VB' in tgch.tag_[:2] for tgch in token.children):
            pass
          elif token.head.tag_[:2] == 'VB':
            pass
          else:
            if token.morph.get('Tense') == ['Pres']:
              ISLTokens.append(nlp('(present)'))
            if token.morph.get('Tense') == ['Past']:
              ISLTokens.append(nlp('(past)'))
        #ISLTokens.pop(done_list[])'''
  for token in doc:
    if token.morph.get('Tense') == ['Past']:
      ISLTokens.append( nlp('(-past-)'))
      break
  #for token in doc:
  #  if 'Tense_past' in nlp.vocab.morphology.tag_map[token.tag_].keys():
  #    ISLTokens.append( nlp('(-past-)'))
  #    break        

    #print([doc[i] for i in done_list])






  if doc2:
    ISLTokens2 = eng_isl_translate(doc2)
    ISLTokens.append(and_tkn)
    ISLTokens.extend(ISLTokens2)

  if doc2_ques:
    ISLTokens2 = eng_isl_translate(doc2_ques)
    ISLTokens.append(ques_tkn)
    ISLTokens.append(nlp("-pause-"))
    ISLTokens.extend(ISLTokens2)


 
  return ISLTokens


def translate_text(text):
  """Convert ISLToken output to space separated gloss list"""
  doc= nlp(text)  
  raw_token_list = eng_isl_translate(doc) #translate_to_tokens(text) #eng_isl_translate(doc)
      
  raw_isl_text = " ".join([isl_tkn.text.lower() for isl_tkn in raw_token_list])

      
  return replace_num(raw_isl_text)


def token_list(text):
  doc= nlp(text) 
  raw_token_list = eng_isl_translate(doc)
  senlist = [replace_num(isl_tkn.text.lower()) for isl_tkn in raw_token_list]
  for i in range(len(senlist)):
    senlist[i] = " " + senlist[i] + " "
    senlist[i] = re.sub('(?:( the | a | an | and | or ) +)',' ',senlist[i])
    senlist[i] = senlist[i].strip()
  return senlist

  #For splitting of each sentence


def input_split(text):
  text = text.strip().lower()
  lines_list = []
  text = re.sub('''[^a-zA-Z0-9 !,;?.'"!():-]''', '', text)
  try:
    if text[-1:] == ',':
      text = text[:-1]
 
    if text[-1] not in ['?','.','!',';']:
      text = text + '.'


    text_len = len(text)
    k=0
    for i in range(text_len):
      try:
        if text[i] in ['.','?','!',';']:
          text[i] == '.'
          lines_list.append(text[k:i+1])
          k=i+1
      except:
        continue


    return lines_list
  except:
    return []


def find_syn(line_list, words_list):
  """Finds a synonym that exists in the available wordlist, from WordNet"""

  fin_list = []
  fin_see_list = []  


  #print(line_list)
  main_sent = nlp(" ".join(line_list))

  main_list = line_list

  new_list = []
  for i in range(len(main_list)):
    #print(new_list , i)
    flag = 0
    #print(main_list[i], main_list)


    if " " in main_list[i]: 
      #print(main_list)
      split_word = main_list[i]
      word_set = split_word.split()
      #print(main_list)
      word_set_count = len(word_set)
      #print(main_list, main_list[i], word_set)

      for m in range(word_set_count,0,-1):
        for n in range(word_set_count+1-m,0,-1):
          #print(n,m, n-1,n-1+m, word_set[n-1:n-1+m])

          if (" ".join(word_set[n-1:n-1+m])) in words_list:
            #main_list = main_list[:i] + word_set[:n] +  word_set[n:n+m] + word_set[n+m:] + main_list[i+1:]
            new_list = new_list + word_set[:n-1] + word_set[n-1:n-1+m]+ word_set[n+m-1:]
            #new_list = new_list + word_set[:n-1] +  word_set[n-1:n-1+m] + word_set[n+m-1:]                  
            word_set_count = len(word_set)
            flag = 1
            break    

      #print(flag, main_list[i])
      if flag == 0:
        temp = nlp(" ".join(word_set))
        temp_set = [tl.lemma_ for tl in temp]
        new_list = new_list + temp_set

      #new_list = main_list[:i] + main_list[i].split() + main_list[i+1:]
        #print(new_list)
 
    else:
      #print(main_list[i] , 1)
      new_list.append(main_list[i])
  
  #print(new_list)
  main_list = new_list
    #if flag == 0:
    #  main_list = main_list[:i] + main_list[i].split() + main_list[i+1:]
    #  new_list = main_list[:i] + main_list[i].split() + main_list[i+1:]






  #print(new_list, "NL" , main_list)
  #print(main_list)
  #Check list is with underscore
  #main_list is the new list
  check_list = []
  for i in main_list:
    check_list.append(i) 


  for word in range(len(main_list)):
    if re.match("[a-z]+-[a-z]+", main_list[word]):
      word_wd_1 = main_list[word].replace('-',' ')
      main_list[word] = word_wd_1
      word_wd_2 = check_list[word].replace('-','_')
      check_list[word] = word_wd_2



  
  #print(main_list, check_list)
  #main_sent will compare tokens with equivalence of sentences
  main_sent = nlp(" ".join(check_list))
  main_sent_wh = nlp(" ".join(main_list))
  main_sent2 = check_list
  


  #Iterate through each word
  for word_spot in range(len(main_list)):
    #print(main_sent[word_spot].tag_)
    if main_list[word_spot] in ['a', 'an' ,'the', 'and','of','at']:
      continue
    #Check if a proper noun
    if main_sent[word_spot].tag_ == "NNP":

      ##LATEST CHANGE 
      if main_sent[word_spot].text in words_list:
        if ' ' in main_list[word_spot]:
          fin_list.append(main_sent[word_spot].text.replace(' ','-'))
        else:
          fin_list.append(main_list[word_spot])
        fin_see_list.append(main_list[word_spot])
      else:
        split_token_text = " ".join([ch for ch in main_list[word_spot]])
        fin_list.append(split_token_text)
        fin_see_list.append(main_list[word_spot])
        ##TILL HERE

    else:
    #But already there in ISL list so go ahead
      if main_list[word_spot] in words_list:
        ##CHANGES
        if ' ' in main_list[word_spot]:
          fin_list.append(main_list[word_spot].replace(' ','-'))
        else:
          fin_list.append(main_list[word_spot])
        fin_see_list.append(main_list[word_spot])
        ##TILL HERE

        #If not in ISL List, see synonyms
      else:
        token_synsets = wordnet.synsets(check_list[word_spot])
        if len(token_synsets) == 0:
          split_token_text = " ".join([ch for ch in main_list[word_spot]])
          fin_list.append(split_token_text)
          fin_see_list.append(main_list[word_spot])
        else:

          simmax = 0.65
          simmax_word = ''
          for synset in token_synsets:
            for word in synset.lemma_names():
              #print( check_list[word_spot], word, token.similarity(nlp("the safe crow drank from jug")[4]))
              

              #If synonym in list
              #print(check_list[word_spot])
              if word in words_list and word!=check_list[word_spot]:
                main_sent2[word_spot] = word

                #simval = nlp(" ".join(sentence_list))[sen_i].similarity(token)
                simval = nlp(" ".join(main_sent2))[word_spot].similarity(main_sent[word_spot])
                #print(simval, word, nlp(" ".join(main_sent2))[word_spot].similarity(main_sent[word_spot]))
                
                if simval > simmax:
                  #print(simval)
                  simmax = simval
                  simmax_word = word

          if simmax_word != '':
            fin_list.append(simmax_word)
            fin_see_list.append(main_list[word_spot])
            fin_see_list.append( "(" + simmax_word + ")" )

          else:
            split_token_text = " ".join([ch for ch in main_list[word_spot]])
            fin_list.append(split_token_text)
            fin_see_list.append(main_list[word_spot])

                
            
  return fin_list, fin_see_list


def final_output(input_split, translate_text ,text,token_list): #,num): #,result_batch):

  output=[]
  output2=[]
  output3=[]
  output_see = '###'
  fin_line = []
  fin_line2 = []
  phrases_temp = ''
  splitted_sentence = [k.strip() for k in input_split(text) if k!= '']

  #Testing
  #splitted_sentence = eng_sent


  #To see words in list
  for line in splitted_sentence:
    line_list = token_list(line)
    #print(line_list)
    #taken of spell check for now
    checked_line = []
    checked_line2 = []



    for token in token_list(line):
      syn_w = find_syn(line_list, words_list)
    
      #What the Unity sees
      for i in syn_w[0]:
          if i in ['( - p a s t - )', '- p o s s i b l e -', '- p a u s e -']:  
              j = i.replace(' ', '')
              syn_w[0][syn_w[0].index(i)]  = j

      for i in syn_w[1]:
          if i in ['(-past-)', '-pause-']:
              j = i.replace(i, '')
              syn_w[1][syn_w[1].index(i)]  = j
          if i in [ '-possible-']:
              j = i.replace(i, 'possible?')
              syn_w[1][syn_w[1].index(i)]  = j
 
    #print(syn_w[0] , syn_w[1])
  
    fin_line =  " ".join(syn_w[0]) + "."
    fin_line2 = " ".join(syn_w[1]) + "."

    '''#TESTING PURPOSE
    ffi.append(fin_line2)
    if num%5==0:
      print(num)
    num+=1
    print(num)'''

    output.append(fin_line)
    #output.append(".")
    #print(output)
    output2.append(fin_line2)
    #output2.append(".")
    #print(output2)
      
  output3 = " ".join(output + [output_see] + output2)

  #testing
  #df_test['ffi model V2'] = ffi

  return output3


text = input("Enter an English sentence \n")
final_view = final_output(input_split, translate_text, text,token_list) #,0)
print(final_view)