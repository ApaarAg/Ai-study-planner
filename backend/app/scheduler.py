import numpy as np
def compute_priority(topic):
    urgency=1/(topic["days_to_exam"]+1)
    gain=topic["predicted_gain"]
    difficulty_factor=topic["difficulty"]/5

    return gain*urgency*(1+0.3*difficulty_factor)

def generate_plan(topics, available_hours):

    for item in topics:
        item["allocated_hours"]=0

    delta=0.25
    total_steps=int(available_hours/delta)

    for _ in range(total_steps):
        best_topic=None
        best_gain=-1

        for item in topics:
            if "predicted_gain" not in item:
                continue
            mg=marginal_gain(
                item["predicted_gain"],
                item["allocated_hours"]
            )
            if np.isnan(mg):
                continue

            if mg>best_gain:
                best_gain=mg
                best_topic=item
        if best_topic is None:
            break
        best_topic["allocated_hours"]+=delta # pyright: ignore[reportOptionalSubscript]

        for item in topics:
            minutes=int(round(item["allocated_hours"]*60))
            item["allocated_minutes"]=minutes
            item["allocated_time"]=f"{minutes//60}h: {minutes%60}m"
        return topics




def total_gain(predicted_gain,hours,k=0.8):
    return predicted_gain*(1-np.exp(-k*hours))

def marginal_gain(predicted_gain,current_hours,delta=0.25,k=0.8):
    before=total_gain(predicted_gain,current_hours,k)
    after=total_gain(predicted_gain,current_hours+delta,k)
    return after-before

