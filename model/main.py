def main():
    # Load dataset
    X_train, X_val, y_train, y_val = load_data()

    # Preprocess data
    X_train_preprocessed, X_val_preprocessed = preprocess_data(X_train, X_val)

    # Define model architecture
    input_size = X_train_preprocessed.shape[1]
    hidden_size = 64
    output_size = len(np.unique(y_train))
    learning_rate = 0.01
    epochs = 100
    patience = 5

    # Train the model
    best_params = train_model(X_train_preprocessed, y_train, X_val_preprocessed, y_val,
                               input_size, hidden_size, output_size, learning_rate, epochs, patience)

    # Evaluate the model
    evaluate_model(best_params, X_val_preprocessed, y_val)

if __name__ == "__main__":
    main()
