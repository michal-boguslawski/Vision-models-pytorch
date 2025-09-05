from evaluation.evaluate import EvaluateProjectModels


if __name__ == "__main__":
    # set directiories
    project_name = "sports_cv_project_classification"
    print(EvaluateProjectModels(project_name=project_name).evaluate())
