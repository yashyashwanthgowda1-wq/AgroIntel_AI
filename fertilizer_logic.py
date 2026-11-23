def recommend_fertilizer(nitrogen, phosphorus, potassium):

    if nitrogen < 50:
        return "Urea Recommended"

    elif phosphorus < 40:
        return "DAP Recommended"

    elif potassium < 40:
        return "MOP Recommended"

    else:
        return "Organic Compost Recommended"