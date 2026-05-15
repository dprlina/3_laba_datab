from flask import Flask, render_template, request

from analysis_engine import load_dataset, validate_dataset, dataframe_summary, build_charts
from llm_agent import generate_report, get_llm_status

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    report = None
    summary = None
    charts = None
    selected_tools = None
    mode = None
    question = "Сделай полный аналитический отчет по датасету heart disease: качество данных, закономерности, важные признаки, baseline-модель и выводы."

    try:
        file = None
        if request.method == "POST":
            question = request.form.get("question", question).strip() or question
            file = request.files.get("dataset")

        df = load_dataset(file_storage=file)
        validate_dataset(df)
        summary = dataframe_summary(df)
        charts = build_charts(df)

        if request.method == "POST":
            result = generate_report(df, question)
            report = result["report"]
            selected_tools = result["selected_tools"]
            mode = result["mode"]
    except Exception as exc:
        error = str(exc)

    llm_status = get_llm_status()

    return render_template(
        "index.html",
        error=error,
        report=report,
        summary=summary,
        charts=charts,
        selected_tools=selected_tools,
        mode=mode,
        question=question,
        llm_status=llm_status,
    )


if __name__ == "__main__":
    app.run(debug=True)
