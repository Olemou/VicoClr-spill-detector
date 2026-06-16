from benchmarking.classifier_metric import evaluate_all_models
from benchmarking.tsne import launch_tsne_data_generation
from benchmarking.utils import  load_setup
def main():
  args = load_setup()
  launch_tsne_data_generation(args)
  _ = evaluate_all_models(args)

if __name__ == "__main__":
    main()
    