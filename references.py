import urllib, urllib.request
import re
from unicode_tr.extras import slugify

def add_authors(paper_id):
  if(re.search('[a-zA-Z]', str(paper_id))):
    aux_id = ""
    first = True
    for i in paper_id:
      if i.isnumeric() and first:
        aux_id += "/" + i
        first = False
      else:
        aux_id += i
    paper_id = aux_id

  url = 'http://export.arxiv.org/api/query?search_query=all:' + paper_id
  data = urllib.request.urlopen(url)
  xml = data.read().decode('utf-8')
  end = xml.find("</title>")
  aux = xml[end-5:end+8]
  xml = re.sub(aux, " ", xml)
  begin = xml.find("<title>")
  end = xml.find("</title>")
  title = xml[begin+7:end]
  title = re.sub("\n", "", title)
  title = re.sub(' +', ' ', title)
  if not title.isascii():
    title = slugify(title)
  output = ""

  while xml.find("<name>")!=-1:
    begin = xml.find("<name>")
    end = xml.find("</name>")
    name = xml[begin+6:end]
    aux_name = name.split(" ")
    for p in range(len(aux_name)):
      if not aux_name[p].isascii():
        aux_word = slugify(aux_name[p])
        if aux_word:
          aux_name[p] = aux_word[0].upper() + aux_word[1:]

    authors = ""
    for i in aux_name[:-1]:
      authors += i + " "
    output += aux_name[-1] + ", " + authors + "and "
    aux = xml[begin:end+6]
    xml = re.sub(aux, ' ', xml)

  published = ""
  begin = xml.find("<published>")
  end = begin + 15
  published = xml[begin+11:end]
  if not published.isascii():
    published = slugify(published)

  output = output[:-5]
  output += " (" + published + "). "  + title + ". " + "https://arxiv.org/abs/" + paper_id

  return output