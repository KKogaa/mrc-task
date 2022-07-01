from datasets import load_dataset
import pandas as pd


def convert_binary(df):
    list_data = []
    for index, row in df.iterrows():
        for index, answer in enumerate(row["answers"]):
            data = {
                "id": row["id"],
                "question": row["question"],
                "context": row["context"],
                "answer": answer,
                "correct": 0,
                "domain": row["domain"],
                "question_type": row["question_type"],
            }
            if index == row["correct_answer_id"]:
                data["correct"] = 1
            list_data.append(data)
    return pd.DataFrame(list_data)


def concatenate_input(df_data):
    df_data["text"] = (
        df_data["question"] + " " + df_data["answer"] + "[SEP]" + df_data["context"]
    )


def get_data_quail():
    df_train = pd.read_csv("quail_train.csv")
    df_val = pd.read_csv("quail_validation.csv")
    df_test = pd.read_json("dev.jsonl", lines=True)
    df_challenge = pd.read_json("challenge.jsonl", lines=True)

    df_train = convert_binary(df_train)
    df_val = convert_binary(df_val)
    df_test = convert_binary(df_test)
    df_challenge = convert_binary(df_challenge)

    concatenate_input(df_train)
    concatenate_input(df_val)
    concatenate_input(df_test)
    concatenate_input(df_challenge)
