#------------------------------------------------------------------#
def NombreBit(x) :
    import numpy as np
    return 1 if x==0 else int(np.log2(x))+1
#------------------------------------------------------------------#
def LireAudioPudup(path) :
    from pydub import AudioSegment 
    return AudioSegment.from_file(path)
#------------------------------------------------------------------#
def ParamAudio(audio) :
    import numpy as np
    data = np.array( np.array(audio.get_array_of_samples()) )
    return data , audio.frame_rate , audio.channels , audio.sample_width
#------------------------------------------------------------------#
def PlageNormalisation(nbr_bit) :
    return (  0  ,  (2**(nbr_bit))-1  )
#------------------------------------------------------------------#
def Normaliser(data,plage,Dtype=None) :
    import numpy as np
    if Dtype == None : Dtype = np.int64
    return np.interp(data,( data.min() , data.max() ),plage).round().astype(Dtype)
#------------------------------------------------------------------#
def DELTA(data):
    import numpy as np
    return np.insert(np.diff(data), 0, data[0])
#------------------------------------------------------------------#
def Traitement(arry,d) :
    import numpy as np
    diff = np.diff(arry)
    diff = np.abs(diff)
    idx = np.concatenate(([0], np.where(diff > d)[0]+1))
    ocu = np.diff(np.concatenate((idx, [arry.size]))) 
    sym = arry[idx]
    return np.repeat(sym,ocu).astype(np.int64)
#------------------------------------------------------------------#
def iDELTA(diff) :
    import numpy as np
    return np.cumsum(diff).astype(diff.dtype)
#------------------------------------------------------------------#
def CreeAuido(data,rate,channels,sampwidth) :
    from pydub import AudioSegment
    return AudioSegment (data.tobytes(),frame_rate=rate,channels=channels,sample_width=sampwidth)
#------------------------------------------------------------------#
def PlayAudio(audio) :
     from pydub.playback import play ;  play(audio)
#------------------------------------------------------------------#
def DictionnaireHuffman(arry): 
    from collections import Counter ; import heapq
    code=[[j,[i,""]] for i,j in Counter(arry).items()]
    heapq.heapify(code)
    while(len(code)>1):                       
        L = heapq.heappop(code) ; R = heapq.heappop(code)  
        for i in R[1:]:i[1]='0'+i[1]  
        for i in L[1:]:i[1]='1'+i[1]
        code.append([R[0]+L[0]]+R[1:]+L[1:])
    return dict(code[0][1:])
#------------------------------------------------------------------#
def HUFFMAN(arry,dictio) :
    return ''.join(map(lambda x: dictio[x], arry))
#------------------------------------------------------------------#
def iHUFFMAN(code,dictio) :
    import numpy as np
    dictio = { dictio[i] : i for i in dictio }
    el = ''
    data = []
    l = len(code)
    for i in range(l) :
        el += code[i]
        if el in dictio : data.append(dictio[el]) ; el =''
    return np.array(data,dtype=np.int64)
#------------------------------------------------------------------#
def Bine(x,nbr) :
    return bin(x)[2:].zfill(nbr)
#------------------------------------------------------------------#
def CodageDictionnaire(dictionnaire) :
    keys   = list(dictionnaire.keys())
    values = list(dictionnaire.values())
    nbrbit_keys = NombreBit(max(keys))
    keys_coder  = ''.join(Bine(x,nbrbit_keys) for x in keys)
    values_string = str(values)[1:-1]
    values_coder = ''.join(Bine(ord(x),6) for x in values_string)
    nbrbit_keys_coder = Bine(nbrbit_keys,8)
    lkeys_coder = Bine(len(keys_coder),32)
    lvalues_coder = Bine(len(values_coder),64)
    return nbrbit_keys_coder+lkeys_coder+lvalues_coder+keys_coder+values_coder
#------------------------------------------------------------------#  
def DecodageDictionnaire(dictBit) :
    nbrbit_keys = int(dictBit[0:8],2)
    lk = int(dictBit[8:40],2)
    lv = int(dictBit[40:104],2)
    keys_coder = dictBit[104:104+lk]
    values_coder = dictBit[104+lk:104+lk+lv]
    keys = [ int(keys_coder[i:i+nbrbit_keys],2) for i in range(0,len(keys_coder),nbrbit_keys)]
    values_str = ''.join([ chr(int(values_coder[i:i+6],2))  for i in range(0,len(values_coder),6)])
    values_str = '['+values_str+']'
    values = eval(values_str)
    return {k:v for k,v in zip(keys,values)}
#------------------------------------------------------------------#
def CodageEntete(rate,channels,samp,plage1,plage2,Dtype,dictionnaire) :
    import numpy as np
    rate = Bine(rate,16)
    channels = '0' if channels==1 else '1'
    samp = '00' if samp == 1 else ('01' if samp==2 else '11' if samp==3 else '10')
    n1 = '0' if plage1[0]<0 else '1'
    plage1 = Bine(abs(int(plage1[0])),16) + Bine(abs(int(plage1[1])),16)
    n2 = '0' if plage2[0]<0 else '1'
    plage2 = Bine(abs(int(plage2[0])),16) + Bine(abs(int(plage2[1])),16)
    Dtype = '00' if Dtype==np.int8 == 1 else ('01' if Dtype==np.int16 else ('11' if Dtype==np.int32 else '10'))
    dictionnaire = CodageDictionnaire(dictionnaire)
    return rate+channels+samp+n1+plage1+n2+plage2+Dtype+dictionnaire
#------------------------------------------------------------------#
def DecodageEntete(entete) :
    import numpy as np
    rate = int(entete[0:16],2)
    channels = 1 if entete[16:17] == '0' else 2
    samp = entete[17:19]
    samp = 1 if samp == '00' else (2 if samp=='01' else 3 if samp=='11' else 4)
    n1 = entete[19:20]
    plage1 = entete[20:52]
    plage1 = [int(plage1[:len(plage1)//2],2),int(plage1[len(plage1)//2:],2)]
    if n1 == '0' : plage1[0]*=-1
    n2 = entete[52:53]
    plage2 = entete[53:85]
    plage2 = [int(plage2[:len(plage2)//2],2),int(plage2[len(plage2)//2:],2)]
    if n2 == '0' : plage2[0]*=-1
    Dtype = entete[85:87]
    Dtype = np.int8 if Dtype=="00" == 1 else (np.int16 if Dtype=="01" else( np.int32 if Dtype=="11" else np.int64))
    dictionnaire = DecodageDictionnaire(entete[87:])
    return rate,channels,samp,tuple(plage1),tuple(plage2),Dtype,dictionnaire
#------------------------------------------------------------------#
def Compression(audio) :
    data , rate , channels , samp = ParamAudio(audio)
    plage_data = data.min() , data.max()
    Dtype = data[0].dtype
    dataN = Normaliser(data,PlageNormalisation(samp*8))
    dataN = Traitement(dataN,25)
    delta = DELTA(dataN)
    plage_delta = delta.min(),delta.max()
    deltaN = Normaliser(delta,PlageNormalisation(NombreBit(delta.max())))
    dictionnaire = DictionnaireHuffman(deltaN)
    bits = HUFFMAN(deltaN,dictionnaire)
    entete = CodageEntete(rate,channels,samp,plage_data,plage_delta,Dtype,dictionnaire)
    TI = len(data)*samp # taille en octet
    TF = (len(bits)+len(entete)+32)/8 # taille en octet
    taux = 1 - (TF/TI)
    print("taux : ",str(taux),"%")
    return bits,entete
#------------------------------------------------------------------#
def Decompression(bits,entete) :
    rate,channels,samp,plage_data,plage_delta,Dtype,dictionnaire = DecodageEntete(entete)
    deltaNR = iHUFFMAN(bits,dictionnaire)
    deltaR = Normaliser(deltaNR,plage_delta)
    dataNR = iDELTA(deltaR)
    dataR = Normaliser(dataNR,plage_data,Dtype)
    audioR = CreeAuido(dataR,rate,channels,samp)
    return audioR
#------------------------------------------------------------------#
def CompressionIRM(path):
    import os
    audio = LireAudioPudup(path)
    newpath = os.path.splitext(path)[0]+'.irm'
    bits,e = Compression(audio)
    taille_entete = Bine(len(e),32)
    with open(newpath,'w') as f :
        f.write(taille_entete+e+bits)
#------------------------------------------------------------------#
def DecompressionIRM(path) :
    import os 
    if os.path.splitext(path)[1] == '.irm' :
        with open(path,'r') as fichier :
            contenu = fichier.read()
        taille_entete = int(contenu[0:32],2)
        entete = contenu[32:32+taille_entete]
        bits = contenu[32+taille_entete:]
        return Decompression(bits,entete)
    else : 
        print("juste les extention irm")
#------------------------------------------------------------------#



