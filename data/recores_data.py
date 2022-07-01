import pandas as pd


class RecoresData:
    def __init__(self, *args, **kwargs):
        self.separator = kwargs["sep"]
        self.df_train = pd.read_csv(kwargs["train"], sep=self.separator)
        self.df_val = pd.read_csv(kwargs["val"], sep=self.separator)
        self.df_test = pd.read_csv(kwargs["test"], sep=self.separator)

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

    def convert_dataframe(self, df: pd.DataFrame, type=None):
        options = ["A", "B", "C", "D", "E"]
        df_list = []
        for index, row in df.iterrows():
            for option in options:

                text = None
                if type == "TRAIN":
                    format = [
                        "QUESTION",
                        "SPACE",
                        "OPTION",
                        "SEP",
                        "REASON",
                        "SPACE",
                        "CONTEXT",
                    ]
                    text = self.concatenate_data(row, format, option)

                if type == "VAL":
                    format = [
                        "QUESTION",
                        "SPACE",
                        "OPTION",
                        "SEP",
                        "REASON",
                        "SPACE",
                        "CONTEXT",
                    ]
                    text = self.concatenate_data(row, format, option)

                if type == "TEST":
                    format = [
                        "QUESTION",
                        "SPACE",
                        "OPTION",
                        "SEP",
                        "REASON",
                        "SPACE",
                        "CONTEXT",
                    ]
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
        self.df_train = self.convert_dataframe(self.df_train, "TRAIN")
        self.df_val = self.convert_dataframe(self.df_val, "VAL")
        self.df_test = self.convert_dataframe(self.df_val, "TEST")

    def get_dataframes(self):
        return (self.df_train, self.df_val, self.df_test)
