import click
from train import prepare_data_and_model, train_model
from evaluate import evaluate_model, print_evaluation_results
from visualize import plot_training_curves, plot_confusion_matrix, visualize_predictions
from config import NUM_EPOCHS
import torch
from sklearn.metrics import confusion_matrix

@click.command()
@click.option('--augment', is_flag=True, help="Use data augmentation during training")
def main(augment):
    model, criterion, optimizer, train_loader, val_loader, dataset = prepare_data_and_model(augment)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, criterion, optimizer, train_loader, val_loader, NUM_EPOCHS
    )

    # Save the model
    torch.save(model.state_dict(), 'dog_breed_classifier.pth')
    print("Model saved successfully.")
    
    # Evaluate the model
    val_preds, val_labels = evaluate_model(model, val_loader)

    # Print evaluation results
    print_evaluation_results(val_labels, val_preds, dataset.classes)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    # Plot confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    plot_confusion_matrix(cm, dataset.classes)

    # Visualize predictions
    visualize_predictions(model, val_loader, dataset)

if __name__ == "__main__":
    main()