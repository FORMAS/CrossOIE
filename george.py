# %%

from polyglot.text import Text

from polyglot.downloader import downloader
downloader.supported_tasks(lang="es")

#%%

# download packages
downloader.download("unipos.es")
downloader.download("embeddings2.es")
downloader.download("pos2.es")


# %%

# def fixExtractions(text):
#    text = text.replace('de el', 'del')
#    return text

# %%

def removePunctuation(text):
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('“', '')
    text = text.replace('”', '')
    return text


# %%

def compareArtigos(a1, a2):
    b1 = a1 in ['el', 'al', 'a']
    b2 = a2 in ['el', 'al', 'a']
    return b1 and b2


# %%

def getIndex(tags, text):
    for i in range(len(tags)):
        # print(tags[i][0], en1[0])
        if tags[i][0] == text[0][0]:
            # print(tags[i])
            for j in range(len(text)):
                if tags[i + j][0] != text[j][0]:  # and not compareArtigos(tags[i+j][0], text[j][0]):
                    break
                # else:
                # print(tags[i+j][0], en1[j])
            if j + 1 == len(text):
                return (i, i + j)


# %%

f = open('Dataset/gamalho/en/extractions-all-labeled.txt')
i = 1
extracoes = []
problemas = []
for l in f:
    ll = l.replace('\n', '').split('\t')
    if len(ll) == 1:
        sentenca = ll[0]
    elif len(ll) == 5:
        e_id = ll[0].strip()
        en1 = ll[1][1:-1].strip()
        fr = ll[2][1:-1].strip()
        en2 = ll[3][1:-1].strip()
        y = ll[4].strip()
        # print(e_id, en1, en2, fr)

        if (sentenca != ''):
            try:
                tripla = {
                    'sentenca': sentenca,
                    'id_extracao': e_id,
                    'en1': en1,
                    'fr': fr,
                    'en2': en2,
                    'y': y,
                    'sentenca_pos': Text(sentenca, 'es').pos_tags,
                    'en1_pos': Text(en1, 'es').pos_tags,
                    'en2_pos': Text(en2, 'es').pos_tags,
                    'fr_pos': Text(fr, 'es').pos_tags,
                }
                if en1 == '0' or en2 == '1':
                    problemas.append(l)
                else:
                    extracoes.append(tripla)
            except ValueError:
                problemas.append(l)
                #raise
    i = i + 1
print(str(i) + ' extracoes lidas')

# %%

# teste
for i in range(len(extracoes)):
    if not getIndex(extracoes[i]['sentenca_pos'], extracoes[i]['en2_pos']) and i not in []:
        print(i)
        break

# %%

pos = extracoes[0]['sentenca_pos']


# %%

def polyglotToText(pos):
    tags = []
    words = []
    for pair in pos:
        tags.append(pair[1])
        words.append(pair[0])

    tags = ' '.join(tags)
    words = ' '.join(words)
    return (words, tags)


# %%

result = []
for extracao in extracoes:
    sentenca = polyglotToText(extracao['sentenca_pos'])
    fn1 = polyglotToText(extracao['en1_pos'])
    fn1_i = getIndex(extracao['sentenca_pos'], extracao['en1_pos'])

    fr = polyglotToText(extracao['fr_pos'])
    fr_i = getIndex(extracao['sentenca_pos'], extracao['fr_pos'])

    fn2 = polyglotToText(extracao['en2_pos'])
    fn2_i = getIndex(extracao['sentenca_pos'], extracao['en2_pos'])

    y = extracao['y']

    linha = '|'.join([sentenca[0], sentenca[1], \
                      fn1[0], fn1[1], str(fn1_i[0]), str(fn1_i[1]), \
                      fr[0], fr[1], str(fr_i[0]), str(fr_i[1]), \
                      fn2[0], fn2[1], str(fn2_i[0]), str(fn2_i[1]), \
                      y])
    result.append(linha)

# %%

f = open('banco-final.csv', 'w')
f.write('sentenca;sentenca_pos;en1;en1_pos;en1_start;en1_end;y\n')
for i in result:
    f.write(i + '\n')


