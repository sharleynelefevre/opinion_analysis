# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 22:21:52 2018

@author: Sharleyne-Lefevre
"""

import codecs
import os
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# fichier d'entrainement (gros volume 70%)
train = codecs.open("C:/Users/Sharleyne-Lefevre/Desktop/cours/Master2/projetTAL/train-test/train_tweets.txt", "r", "utf8")

# fichier pour le test (moins volumineux 30%)
test = codecs.open("C:/Users/Sharleyne-Lefevre/Desktop/cours/Master2/projetTAL/train-test/test_tweets.txt", "r", "utf8")

# fichiers pour la lemmatisation
tempinFilename = "C:/Users/Sharleyne-Lefevre/Desktop/cours/Master2/projetTAL/train-test/tempin.txt"
tempoutFilename = "C:/Users/Sharleyne-Lefevre/Desktop/cours/Master2/projetTAL/train-test/tempout.txt"


def recupTweets(file):
    file = file.readlines()
    # ajout d'une ligne vide pour pouvoir récupérer le dernier tweet sinon il n'était pas récupéré
    file.append('')
    output = []
    tweet = []
    # pour chaque ligne dans la range du nombre de lignes
    for i in range(len(file)):
        line = file[i].split()
#        print(len(line)) # nb d'éléments dans la ligne
        # si le nombre d'éléments dans la ligne est égal à 0 -> ligne vide
        # et si la ligne n'est pas vide : []
        if len(line) == 0 and tweet != []:
            # on ajoute à la liste output le tweet précédemment ajouté dans la liste tweet 
            # après le premier tour de boucle
            output.append(tweet)
            # à chaque fois qu'on tombe sur 0 on créé une liste vide pour y mettre le futur tweet
            tweet = []
        else:
            # si le nombre d'élément dans la ligne est supérieur à 1 élément
            # comme ça on n'ajoute pas l'identifiant qui est égal à 1 élément dans la ligne
            if len(line) > 1: 
                # on ajoute un tuple avec l'élément 0 (mot) et l'élément 3 (label) de chaque ligne du tweet dans la liste tweet
                tweet.append((line[0], line[3]))
#                print(tweet)
                
    # on ouvre tempinFilename en écriture
    f_in = codecs.open(tempinFilename, "w", "utf8")
    # pour écrire chaque mot dans le fichier tempinFilename
    for tweet in output:
        # pour chaque tuple dans tweet
        for tupl in tweet:
            # on écrit l'élément 0 de chaque tuple dans le fichier tempinFilename
            f_in.write(str(tupl[0]) + "\n")
    return output


def traitement(data):
    print("ETAPE 1 : Récuperation des données")
    # res est une liste de listes
    # les listes dans res sont composées de tuples qui forment le tweet
    # pour chaque mot du tweet on a un tuple de deux éléments ("mot", "label")
    # res contient ce que retourne la fonction recupTweets, du fichier mis dans le parametre data (fichier d'entrainement)
    res = recupTweets(data)
#    print(res)
    print("ETAPE 1 finie")


    print("ETAPE 2 : Lemmatisation des données")
    # tagging des mots avec TreeTagger
    os.system("d:/Tools/TreeTagger/bin/tag-french.bat " + tempinFilename + " " + tempoutFilename)
    
    # ouverture du fichier de sortie en lecture qui contient le tagging des mots par TreeTagger
    f_out = open(tempoutFilename, "r", encoding='utf-8')
    
    # initialisation d'une liste qui accueillera la catégorie grammaticale et le lemme du mot
    lemma_tab = []
    for lemme in f_out:
#        print(lemme)
#        print(lemme.split()[0])
        if lemme.split()[0] :
            #ALM : l'utilisation d'une liste permet de pouvoir récupérer les lemmes et les PoS dans l'ordre des mots du texte. 
            lemma_tab.append((str(lemme.split()[1]),str(lemme.split()[2])))
#    print(lemma_tab)
    print("ETAPE 2 finie")


    print("ETAPE 3 : Création des traits")
   # output contient chaque tweet dans une liste, dans chaque liste il y a autant de tuples que de mots dans le tweet
    output = []
    #ALM : la variable j permet de connaitre l'index de la phrase dans res, la variable i est l'indice des tokens dans la phrase et la variable k est l'indice des tokens dans le corpus complet.  
    j = 0
    k = 0
    # pour chaque tweet dans res
    for tweet in res:
        i = 0
        # on créé une liste
        tweetUnite = []
        # pour chaque mot dans le tweet
        for mot in tweet:
            mot_str = mot[0]
            # on créé un dictionnaire
            feature = {}
            # on met dans le dictionnaire le mot en minuscule
            feature["word.lower()"]= mot_str.lower()
            # le mot en maj
            feature["word.isupper()"] = mot_str.isupper()
            # un booléen qui renvoit vrai si le mot commence par une majuscule
            feature["word.istitle()"] = mot_str.istitle()
            # la categorie grammaticale du mot
            #ALM : récupération du postag 
            feature["postag"] = lemma_tab[k][0]
            # et le lemme du mot
            feature["lemme"] = lemma_tab[k][1] 
            # on ajoute le dictionnaire dans la liste tweetTemp
            tweetUnite.append(feature)
            #ALM : ajout de l'élément PoS entre le token et le label
            m = list(mot)
            m.insert(1,feature["postag"])
            mot = tuple(m)
            r = list(res[j])
            r[i] = mot
            res[j] = tuple(r)
            i+= 1
            k+= 1
        j+=1
        # on ajoute le tout dans la liste output    
        output.append(tweetUnite)    

    # ouverture du fichier csv en lecture contenant les termes porteurs de sentiments
    with codecs.open("C:/Users/Sharleyne-Lefevre/Desktop/cours/Master2/projetTAL/train-test/FEEL.csv", 'r', "utf8") as csvfile:
    
        for row in csvfile:
            row = row.split(";")
            # les mots porteurs de sentiment du csv se trouvent en row[1]
            currentWordCSV = row[1]
            # retire le retour à la ligne de la dernière colonne
            row[len(row)-1] = row[len(row)-1].replace("\r\n","")
            
            for tweet in output:
                for mot in tweet:
                    # on se base sur le lemme pour la comparaison
                    currentWordFile = mot["lemme"]
                    # si le mot courant dans le tweet est le même qu'un mot de la liste de termes porteurs de sentiment
                    if currentWordCSV == currentWordFile:
                        # on ajoute au dictionnaire la polarité
                        mot["polarity"] = row[2]
                        # la polarité joy
                        mot["joy"] = row[3]
                        # la polarité fear
                        mot["fear"] = row[4]
                        # la polarité sadness
                        mot["sadness"] = row[5]
                        # la polarité anger
                        mot["anger"] = row[6]
                        # la polarité surprise
                        mot["surprise"] = row[7]
                        # la polarité disgust
                        mot["disgust"] = row[8]
                
    print("ETAPE 3 finie") 
    return (res, output) # on retourne res (tuple de 3 éléments -> mot + catgr + label) et output (les traits de chaque mot)


# cette fonction permet d'appliquer le même traitement pour chaque fichier (train et test) sans redondance
def preApprentissage(fichier):
    sents = traitement(fichier)
    
    print("ETAPE 4 : Définition des traits et labels")
    # comme dans le tuto mais adapté avec mes traits
    # X pour les traits de chaque mot de chaque tweet 
    X = sents[1] # sents[1] = output 
    
    # pour n'avoir que les labels de chaque mot de chaque tweet
    labelsOnly = []
    for tweet in sents[0]: # sents[0] = res
        labelsTweet = []
        for mot in tweet:
            # dans res on prend le 3e (position 2) element de chaque tuple dans chaque tweet (label)
            labelsTweet.append(mot[2]) 
        # on met labelsTweet dans labelsOnly pour avoir un format comme dans le tutoriel    
        labelsOnly.append(labelsTweet) 
    
    Y = labelsOnly # Y pour uniquement les labels (IOB)
    print("ETAPE 4 finie")
    print("Traitement de ce fichier terminé")
    print("-----------------------------")
    return (X, Y)

# traits / labels pour chaque fichier
trainData = preApprentissage(train) 
testData = preApprentissage(test) 

# comme dans le tuto
X_train = trainData[0] #outuput (traits)
y_train = trainData[1] #res (labels)
X_test = testData[0]
y_test = testData[1]
    

# à partir d'ici, rien n'a été modifié
print("ETAPE 5 : Entrainement")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
print("ETAPE 5 finie")


print("ETAPE 6 : Evaluation")
labels = list(crf.classes_)
labels.remove('O')
#print(labels)

y_pred = crf.predict(X_test)
(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))

### résultat par classe ###
# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

scores = (metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))
print("ETAPE 6 finie")

print(scores)