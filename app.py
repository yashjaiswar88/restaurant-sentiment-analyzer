import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr


# ============================================================
# 1. Load trained pipeline
# ============================================================

MODEL_PATH = "restaurant_sentiment_pipeline.joblib"

pipeline = joblib.load(MODEL_PATH)

print("✅ Trained model loaded successfully!")


# ============================================================
# 2. Example reviews
# ============================================================

example_reviews = {
    "😊 Very Positive":
        "The food was absolutely delicious and "
        "the staff were wonderful. I loved everything!",

    "😐 Mixed":
        "The food was okay, but the service was slow "
        "and the overall experience was average.",

    "😡 Very Negative":
        "The food was terrible and the service was "
        "extremely disappointing. I would not recommend this restaurant."
}


# ============================================================
# 3. Main analysis function
# ============================================================

def analyze_input(text):

    empty_table = pd.DataFrame(
        columns=[
            "Review",
            "Sentiment",
            "Confidence"
        ]
    )

    if not text or not text.strip():

        return (
            "Please enter a review.",
            "Confidence: --",
            empty_table,
            None
        )

    # One non-empty line = one review
    reviews = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


    # ========================================================
    # SINGLE REVIEW
    # ========================================================

    if len(reviews) == 1:

        review = reviews[0]

        prediction = pipeline.predict(
            [review]
        )[0]

        probabilities = pipeline.predict_proba(
            [review]
        )[0]

        classes = pipeline.named_steps[
            "classifier"
        ].classes_

        prediction_index = list(
            classes
        ).index(prediction)

        confidence = (
            probabilities[prediction_index] * 100
        )

        sentiment = (
            "Positive"
            if prediction == 1
            else "Negative"
        )

        return (
            sentiment.upper(),
            f"Confidence: {confidence:.2f}%",
            empty_table,
            None
        )


    # ========================================================
    # BATCH ANALYSIS
    # ========================================================

    predictions = pipeline.predict(reviews)

    probabilities = pipeline.predict_proba(reviews)

    classes = pipeline.named_steps[
        "classifier"
    ].classes_

    results = []

    for review, prediction, probability in zip(
        reviews,
        predictions,
        probabilities
    ):

        prediction_index = list(
            classes
        ).index(prediction)

        confidence = (
            probability[prediction_index] * 100
        )

        sentiment = (
            "Positive"
            if prediction == 1
            else "Negative"
        )

        results.append({
            "Review": review,
            "Sentiment": sentiment,
            "Confidence": f"{confidence:.2f}%"
        })

    results_df = pd.DataFrame(results)


    # ========================================================
    # VISUALIZATION
    # ========================================================

    counts = (
        results_df["Sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Negative"],
            fill_value=0
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Sentiment Distribution"
    )

    ax.set_xlabel(
        "Sentiment"
    )

    ax.set_ylabel(
        "Number of Reviews"
    )

    plt.xticks(rotation=0)

    plt.tight_layout()


    return (
        "BATCH ANALYSIS COMPLETED",
        "",
        results_df,
        fig
    )


# ============================================================
# 4. Example loader
# ============================================================

def load_example(name):

    return example_reviews.get(
        name,
        ""
    )


# ============================================================
# 5. Build GUI
# ============================================================

with gr.Blocks(
    title="Restaurant Sentiment Analyzer"
) as demo:

    gr.Markdown(
        """
        # 🍴 Restaurant Review Sentiment Analyzer

        Enter **one review** for individual analysis or
        **multiple reviews, one per line**, for batch analysis.
        """
    )


    # --------------------------------------------------------
    # Common input
    # --------------------------------------------------------

    review_input = gr.Textbox(
        label="Enter Review(s)",
        placeholder=(
            "One review = individual analysis\n"
            "Multiple reviews = one review per line"
        ),
        lines=8
    )


    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    analyze_button = gr.Button(
        "🔍 Analyze",
        variant="primary"
    )


    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    sentiment_output = gr.Textbox(
        label="Sentiment",
        interactive=False
    )

    confidence_output = gr.Textbox(
        label="Prediction Confidence",
        interactive=False
    )


    # --------------------------------------------------------
    # Example buttons
    # --------------------------------------------------------

    gr.Markdown(
        "## Quick Example Reviews"
    )

    with gr.Row():

        positive_button = gr.Button(
            "😊 Very Positive"
        )

        mixed_button = gr.Button(
            "😐 Mixed"
        )

        negative_button = gr.Button(
            "😡 Very Negative"
        )


    positive_button.click(
        lambda: load_example(
            "😊 Very Positive"
        ),
        outputs=review_input
    )

    mixed_button.click(
        lambda: load_example(
            "😐 Mixed"
        ),
        outputs=review_input
    )

    negative_button.click(
        lambda: load_example(
            "😡 Very Negative"
        ),
        outputs=review_input
    )


    # --------------------------------------------------------
    # Batch results
    # --------------------------------------------------------

    gr.Markdown(
        "## Batch Analysis Results"
    )

    batch_output = gr.Dataframe(
        headers=[
            "Review",
            "Sentiment",
            "Confidence"
        ],
        interactive=False
    )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    gr.Markdown(
        "## Sentiment Visualization"
    )

    sentiment_plot = gr.Plot()


    # --------------------------------------------------------
    # Connect Analyze button
    # --------------------------------------------------------

    analyze_button.click(
        analyze_input,
        inputs=review_input,
        outputs=[
            sentiment_output,
            confidence_output,
            batch_output,
            sentiment_plot
        ]
    )


# ============================================================
# 6. Launch application
# ============================================================

if __name__ == "__main__":

    print("✅ Model loaded.")
    print("✅ GUI ready.")

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 10000)
        )
    )