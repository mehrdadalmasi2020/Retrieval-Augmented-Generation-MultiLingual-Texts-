path='/mnt/data/'
read=open(path+"/try/"+"test.txt","r",encoding='utf-8')
wholeText=""
for e in read:
    wholeText=wholeText+e

QApart=wholeText.split("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

Flag=False
CleanText=""
for e in QApart[1].split("\n"):
    if "?" in e or Flag==True:
        CleanText=CleanText+e
        Flag=False
        CleanText=CleanText+e
        print(e)
    if "?" in e :
        Flag=True
