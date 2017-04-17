import sys
args = sys.argv
#Read file
txt = open(args[1],'rt')
txtall = txt.read()
txt.close()

#separate txt
septxt = txtall.split(' ')


dic ={}
txtc = len(septxt) -1
i = 0
for i in range(txtc):
    settxt = septxt[i].strip()  #単語の両端の空白，改行，タブ文字の削除
    judge = settxt[0].isalpha() #1文字目が英語か判定
    if judge :
        if settxt in dic:
            dic[settxt] += 1
        else:
            dic[settxt] = 1
i = 0

for k, v in sorted(dic.items() , key=lambda x:x[1] , reverse=True):
    print (k, v)

