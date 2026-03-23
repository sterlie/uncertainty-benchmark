

private notes to understand what is going on: 

Set up: 
1) Train model on clean data. 
    Datasets: mnist, chest X-ray (CheXpert, NIH, VinDr-CXR), dermatology (MILK10k). 
2) Apply UQ method: 
    probibalistic methods: MC dropout, deep ensemble, SWAG, laplace, test-time augmentation, HETXL. 
    Deterministic: Entropy, DDU. 
2) Create contolled condition where uncertainty is expected to increase.  

To test Aletoric uncertainty: 
MNIST -> data distortions 
chest xray -> young/old, patient have more dieseases . 
Dermatology -> age group, underrepresetned skintones. 

To test Epistemic uncertainty: 
MNIST -> blur images.
chest xray -> radiologist disagreement
Dermatology -> poor image quality, comments of 'gel', 'water drop', or 'dermoscopy liquid'. 



Disentanflement of uncertainty: AUROC curve comparison. 