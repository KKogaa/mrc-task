import pandas as pd


class RecoresData:
    def __init__(self, *args, **kwargs):
        self.separator = kwargs["sep"]
        self.df_train = pd.read_csv("data/" + kwargs["train"], sep=self.separator)
        self.df_val = pd.read_csv("data/" + kwargs["val"], sep=self.separator)
        self.df_test = pd.read_csv("data/" + kwargs["test"], sep=self.separator)

    def concatenate_data(self, row, format, option):
        """
        Concatenates data into the text from the row given the specified order in the format
        Fomat tokens:
        QUESTION
        OPTION
        REASON
        CONTEXT
        MODIFIED_CONTEXT
        SEP
        SPACE
        """
        text = ""
        for token in format:
            if token == "QUESTION":
                text = text + row["question"]

            if token == "OPTION":
                text = text + row[option]

            if token == "REASON":
                text = text + row["reason"]

            if token == "CONTEXT":
                text = text + row["context"]

            if token == "SEP":
                text = text + "[SEP]"

            if token == "SPACE":
                text = text + " "

            if token == "MODIFIED_CONTEXT":
                text = text + row["modified_context"]

        return text

    def convert_dataframe(
        self,
        df: pd.DataFrame,
        format,
        type=None,
    ):
        options = ["A", "B", "C", "D", "E"]
        df_list = []
        for index, row in df.iterrows():
            for option in options:

                text = None
                if type == "TRAIN":
                    text = self.concatenate_data(row, format, option)

                if type == "VAL":
                    text = self.concatenate_data(row, format, option)

                if type == "TEST":
                    text = self.concatenate_data(row, format, option)

                data = {
                    "text": text,
                    "question": row["question"],
                    "answer": row[option],
                    "context": row["context"],
                    "reason": row["reason"],
                }

                if option == row["answer"]:
                    data["correct"] = 1
                else:
                    data["correct"] = 0

                df_list.append(data)

        return pd.DataFrame(df_list)

    def setup(self):
        self.df_train.rename(columns={"text": "context"}, inplace=True)
        self.df_val.rename(columns={"text": "context"}, inplace=True)
        self.df_test.rename(columns={"text": "context"}, inplace=True)

        format = [
            "QUESTION",
            "SPACE",
            "OPTION",
            "SEP",
            "SPACE",
            "CONTEXT",
        ]
        self.df_train = self.convert_dataframe(self.df_train, format, "TRAIN")

        self.df_val = self.convert_dataframe(self.df_val, format, "VAL")

        format = [
            "QUESTION",
            "SPACE",
            "OPTION",
            "SEP",
            "SPACE",
            "CONTEXT",
        ]
        self.df_test = self.convert_dataframe(self.df_test, format, "TEST")

    def get_dataframes(self):
        return (self.df_train, self.df_val, self.df_test)
