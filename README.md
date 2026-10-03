# Pattern Recognition, final project

Final project of the Pattern Recognition course at the Democritus University of Thrace (Fall 2023 / early 2024). It has two parts, an image classifier with anomaly detection and a binary classifier for drug discovery.

| Part | Task | Method | Result |
|---|---|---|---|
| 1 | Faces with and without a mask, and detection of a class never seen in training (mask worn incorrectly) | CNN in Keras tuned with Keras Tuner (Hyperband), and a separate dense network that gives the anomaly score | 97.13% test accuracy. 91.07% of the 56 incorrect-use images flagged as anomalies |
| 2 | Whether a molecule binds to a target receptor | Standardized continuous features, PCA, and a 1D CNN tuned with Keras Tuner (RandomSearch) | AUC 0.9509 and 90.58% accuracy on the validation split |

The report (`report/Report_Final_Project.pdf`) and the slides (`report/slides.pptx`) are in Greek.

## Part 1, face masks

The data (`Mask_DB.zip`, provided with the assignment) has 1,044 images with a mask, 1,044 without and 56 with the mask worn incorrectly. The third class is kept out of training.

1. The two main classes are split 60/20/20 into training, validation and test sets.
2. A CNN with three convolutional layers is tuned with Keras Tuner (Hyperband over the number of filters and the size of the dense layer), with early stopping on the validation loss (patience 3).
3. The best model reaches 97.13% accuracy on the test set.
4. For the unseen class, a second, fully connected network with a sigmoid output is trained on the two main classes. An incorrect-use image counts as an anomaly when its score is above 0.0005. With this threshold 91.07% of the 56 images are flagged and the other 8.93% are taken for "with mask". As the report notes, such a low threshold can also raise the false positives.

The report also discusses data augmentation, class-dependent costs and transfer learning, and why anomaly detection was chosen instead.

Code: [`part1-mask-classifier/notebook.ipynb`](part1-mask-classifier/notebook.ipynb)

## Part 2, drug discovery

The data (`Data_Receptors.zip`, provided with the assignment) describes each molecule with 3,473 features, 1,425 continuous descriptors followed by 2,048 binary ones. The label says whether the molecule binds to the receptor.

1. The continuous features are standardized and joined again with the binary ones.
2. PCA keeps the components that explain 95% of the variance.
3. The training molecules are split 80/20. A 1D CNN (one to three convolutional layers, a dense layer and dropout) is tuned with Keras Tuner (RandomSearch, 15 trials) on the 80%, with the 20% as validation.
4. On the 20% the best model reaches 90.58% accuracy and AUC 0.9509.
5. The model then predicts the test molecules. [`test_predictions.csv`](part2-drug-discovery/test_predictions.csv) has the predicted label and the score for each one.

Code: [`part2-drug-discovery/notebook.ipynb`](part2-drug-discovery/notebook.ipynb)

## Repository structure

```
.
├── report/
│   ├── Report_Final_Project.pdf
│   └── slides.pptx
├── part1-mask-classifier/
│   └── notebook.ipynb
└── part2-drug-discovery/
    ├── notebook.ipynb
    └── test_predictions.csv
```

## Running the code

The notebooks were run on Google Colab and read `Mask_DB.zip` and `Data_Receptors.zip` from `/content`. The datasets were provided with the course and are not included here.

```bash
pip install jupyter numpy pandas matplotlib scikit-learn opencv-python tensorflow keras-tuner
```

## License

MIT, see [LICENSE](LICENSE).

## Author

[Dimitrios Anastasoudis](https://github.com/anastasoudis), [LinkedIn](https://linkedin.com/in/anastasoudis)
