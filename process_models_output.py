

def ProcessCaption(caption):
    if(caption is None or caption == ""):
        return "nocaption"

    captionWords = caption.split()
    finalCaption = []
    if(captionWords[-1]=="eeee"):
        for i in range(0,len(captionWords)-1):
            finalCaption.append(captionWords[i])

    returncap = " ".join(finalCaption)
    return returncap


def ProcessTags(tags):
    if(tags is None or tags == ""):
        return "notags"

    tagsArray = tags.split()
    tagsDict = {i:tagsArray.count(i) for i in tagsArray}
    finalTag = ""

    for key, value in tagsDict.items():
        if(value > 1):
            lastLetter = key[-1:]
            last2Letters = key[-2:]
            if(lastLetter == "s" or lastLetter == "x" or lastLetter == "z" or last2Letters == "sh" or last2Letters == "ch"):
                finalTag += str(value)+" "+key+"es, "
            else:
                finalTag += str(value)+" "+key+"s, "
        else:
            finalTag += key+", "

    return finalTag