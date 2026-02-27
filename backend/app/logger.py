import csv,os,datetime
import pandas as pd

BASE_DIR= os.path.dirname(os.path.dirname(__file__))
DATA_DIR=os.path.join(BASE_DIR,"data")
os.makedirs(DATA_DIR,exist_ok=True)

LOG_PATH=os.path.join(DATA_DIR,r"session_logs.csv")

def log_session(predicted_plan,final_plan):
    
    file_exits=os.path.isfile(LOG_PATH)

    final_lookup={
        item.topic_name:item.allocated_minutes
        for item in final_plan
    }

    with open(LOG_PATH,mode="a",newline="") as file:
        writer=csv.writer(file)

        if not file_exits:
            writer.writerow([
                "timestamp",
                "topic_name",
                "difficulty",
                "past_score",
                "hours_spent",
                "revision_count",
                "days_to_exam",
                "confidence",
                "predicted_gain",
                "predicted_minutes",
                "final_minutes",
                "delta_minutes"
            ])

            timestamp=datetime.datetime.now()

            for topic in predicted_plan:

                predicted_minutes=topic["allocated_minutes"]
                final_minutes=final_lookup.get(topic["topic_name"],predicted_minutes)
                delta=final_minutes-predicted_minutes

                writer.writerow([
                    timestamp,
                    topic["topic_name"],
                    topic["difficulty"],
                    topic["past_score"],
                    topic["hours_spent"],
                    topic["revision_count"],
                    topic["days_to_exam"],
                    topic["confidence"],
                    topic["predicted_gain"],
                    predicted_minutes,
                    final_minutes,
                    delta
                ])
                normalized_gain=compute_normalized_gain(
                    topic["pre_score"],
                    topic["post_score"]
                )
            if os.path.isfile(LOG_PATH):
                df=pd.read_csv(LOG_PATH)
                if len(df)>=300:
                    print("Retraining threshold reached (300 logs)")

def compute_normalized_gain(pre_score,post_score):
    if pre_score>=100:
        return 0.0
    return (post_score-pre_score)/(100-pre_score)

