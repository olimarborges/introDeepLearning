# cars_feature_extractor

- Colocar as imagens no diretório "/home/ml/datasets/DeepLearningFiles": Peguei as 173 imagens originais
- Executar o script augmentate_new.py: Deixei as imagens do mesmo tamanho (224,224) e realizei augmentate, resultando em 1.903 imagens no total, contando com as originais
- Executar o script extract_features_new.py: Extraí as features das 1.903 imagens utilizando o modelo 'model.hdf5', deixando como chave do dicionário, o nome da imagem. Ao final, é gerado o arquivo 'cars.npz'
- Executar o script load_features_new.py: Apresenta as informações das features geradas no passo anterior
- Executar o script pre-process_new.py: Separa as 1.903 imagens totais em duas pastas, 'train'(1522-80%),'test'(381-20%)
- Executar o script train_and_test_new.py: Realiza o treinamento com as features, apresentando a acurácia de 7 modelos diferentes (Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Trees, Random Forests, Gaussian Naive Bayes e Support Vector Machine). Em seguida, precisa alterar no script para qual modelo deseja verificar o melhor classificar. Após escolhido, basta comentar o trecho que apresenta as acurácias e executar novamente o script, que fará a predição. São selecionadas 20 imagens aleatórias das imagens de teste e apresentado de forma visual qual foi a predição para a imagem