import datasets

if __name__ == "__main__":
    print("obtaining mega dataset")

    # obtaining RACE + QUAIL + RECORES datasets and concatenating
    
    quail = datasets.load_dataset("quail", None)

    race = datasets.load_dataset("race", None)

    #TODO: create binary megadaset
    #TODO: create multi megadaset