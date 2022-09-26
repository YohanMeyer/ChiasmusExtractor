from chiasmus import ChiasmusDetector
import sys


def main():
    fileName = sys.argv[1]

    print('initialize detector')
    chidect = ChiasmusDetector(
            fasttext_model = './fasttext_models/wiki.en.bin',
            feature_types = ['dubremetz', 'lexical', 'embedding'],
            conjlist = ["and", "so", "because", "neither", "nor", "but", "for", "yet"],
            neglist = ["no", "not", "never", "nothing"],
            pos_blacklist=["SPACE", "PUNCT", "PROPN", "DET"],
            spacy_model = 'en_core_web_lg'
            )

    # print('train with crossvalidation')
    # chidect.train_with_crossval(
    #         training_file='data_example/data.json',
    #         num_runs=5
    #         )
    # 
    # chidect.print_summary()

    print('train on whole dataset')
    chidect.train(
            training_file='data_example/data.json', 
            keep_model=True
            )

    print('find chiasmi in new text')
    chidect.run_pipeline_on_text(
            filename=f'{fileName}.txt', 
            text_folder="data_example",
            processed_folder="processed",
            candidates_folder="candidates",
            id_start="test_"
            )

    print('get top candidates')
    chidect.get_top(
            f'candidates/{fileName}.txt.pkl', 
            f'{fileName}-results.json',
            100)

if __name__ == "__main__":
    main()
