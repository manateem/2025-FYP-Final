- [x] Combine manual annotation datasets (Laci)
- [x] 2 Column Template for Overleaf
- [x] Split up Overleaf into multiple .tex files with only one main
- [x] Figure out how to get Overleaf to push to a separate github
- [x] Everyone installs Zotero for citations; + group set up
- [x] Split dataset into training data and testing data and DO NOT TOUCH TESTING DATA until we have a model to use
- [x] submit something for the optional hand-in (latest 23/04) 

- [ ] ~~Catch up on all exercises if not already~~
- [x] Write into Overleaf an introduction section of our goal and the background on the problem/issue
    - [x] Research the background of the problem to inform ourselves and write about it
- [ ] Remove duplicates
- [ ] Segmentation of skin lesions 
    - Use previous students' masks for lesion extraction
    - [ ] Hair extraction
        - [x] Account for different hair colors?
        - [x] Account for amount of hair
        - [x] Utilize hair amount annotations to see how well our hair feature extraction performs
        - [ ] Update data exploration section in Overleaf to reflect changes made, if any
- [ ] Classification
    - [ ] ABC Classifier
    - [ ] Extended Classifier
    - Open Question
        - [ ] Find out what features we want to extract for mega-classifier:
            - [ ] Metadata
                - What to do about missing metadata? Different classifier?
            - [ ] !Hair
            - [ ] !ABC
            - [ ] Random Noise Features
            - [ ] 'Stupid' Features (i.e. top right pixel color)
            - [ ] ...?
        - [ ] Train and create classifier, see performance
        - [ ] See which features are insignificant and remove features and retrain
        - [ ] Repeat as needed
    
    - [ ] Test all our classifiers!!!

Reminders:
- [ ] Acknowledgements to students from other years for extraction Python code
- [ ] Citations for what "Asymmetry, Border, Color" means in the context of skin lesions
