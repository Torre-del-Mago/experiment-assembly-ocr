
def compute_cer(pred_ids, label_ids, processor, cer_metric):
    """
    Compute the Character Error Rate (CER) for predictions and labels.

    Args:
        pred_ids (torch.Tensor): Predicted token IDs from the model.
        label_ids (torch.Tensor): Ground truth token IDs.
        processor (TrOCRProcessor): Processor to decode token IDs into text.
        cer_metric (evaluate.Metric): CER metric object from the `evaluate` library.

    Returns:
        float: The CER score for the batch.
    """
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    # Decode predictions and labels into text
    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute CER using the evaluate library
    cer = cer_metric.compute(predictions=pred_texts, references=label_texts)
    return cer