from recommender import Recommender


r = Recommender("./datasets/items.csv","./datasets/users.csv","./datasets/events.csv")

r.analyse()

r.train()

#Evaluate recommender
with open("./sessions.csv",'r')as f:
    hits = 0
    total = 0
    for row in f.readlines():
        parts = row.split("\t")
        session = parts[0].split(",")
        target_item = row[1]
        recommended = r.recommend(session)
        if target_item in recommended:
            hits +=1
        total+=1
    print("Hits: {}/{}".format(hits, total))