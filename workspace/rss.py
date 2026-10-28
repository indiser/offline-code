import feedparser

url = 'http://feeds.bbci.co.uk/news/rss.xml'
d = feedparser.parse(url)

for entry in d.entries[:3]:
    print(entry.title)