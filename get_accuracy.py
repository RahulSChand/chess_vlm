import numpy as np
import argparse
import pickle

def eval_strict(out, board_pos):

    #! Only reward if non zero entries match. First find the total non zero entries in board_pos (which is ground truth).
    # Create a boolean mask for non-zero elements in board_pos
    non_zero_mask = board_pos != 0

    # If there are no non-zero elements in board_pos, return 1.0 (or handle as appropriate)
    if not np.any(non_zero_mask):
        return 1.0  # Or 0.0, or raise an error, depending on desired behavior

    # Compare elements of out and board_pos where board_pos is non-zero
    correct_matches = out[non_zero_mask] == board_pos[non_zero_mask]

    # Calculate accuracy
    accuracy = np.mean(correct_matches)
    
    return accuracy

def eval_describe_board(args):

    # Load predictions
    predictions = np.load(args.processed_output)
    predictions = predictions[:args.num_samples]

    total_accuracy = 0
    for i in range(args.num_samples):
        board_pos_path = f"{args.test_dataset}/board_pos_{i}.npy"
        board_pos = np.load(board_pos_path)
        total_accuracy += eval_strict(predictions[i], board_pos)
    
    average_accuracy = total_accuracy / args.num_samples
    print(f"Average accuracy: {average_accuracy:.4f}")


def eval_best_move(args):

    with open(f'{args.test_dataset}/best_moves.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    with open(f'{args.test_dataset}/color.pkl', 'rb') as f:
        color_data = pickle.load(f)


    with open(args.processed_output, 'rb') as f:
        filtered_data = pickle.load(f)


    new_filtered_data = filtered_data[:-1]


    k_values = [1, 3, 5, 10]

    for k in k_values:
        correct = 0
        for i in range(len(new_filtered_data)):
            my_prediction = new_filtered_data[i]
            gt_preds = loaded_data[i]

            # print(gt_preds[:2])
            if my_prediction in gt_preds[:k]:
                correct += 1

        print(f"Accuracy for top-k={k}: {correct/len(new_filtered_data):.4f}")




def main():
    parser = argparse.ArgumentParser(description='Evaluate chess board position predictions')
    parser.add_argument('--processed_output', type=str, required=True,
                      help='Path to the processed output numpy file')
    parser.add_argument('--test_dataset', default="dataset_random_384/", type=str,
                      help='Path to the test dataset folder')
    parser.add_argument('--num_samples', type=int, default=128,
                      help='Number of samples to evaluate')
    
    parser.add_argument('--task', type=str, default="describe_board", help='Task to evaluate on', choices=["describe_board", "best_move"])

    args = parser.parse_args()

    if args.task == "describe_board":
        eval_describe_board(args)
    elif args.task == "best_move":
        eval_best_move(args)
    else:
        raise ValueError(f"Invalid task: {args.task}")


if __name__ == "__main__":
    main()  







