def isl_features(hands):
    ordered = sorted(hands, key=lambda hand: hand["lmList"][0][0])[:2]
    if not ordered:
        return None
    origin = ordered[0]["lmList"][0]
    features = []
    for hand in ordered:
        for point in hand["lmList"]:
            features.extend((point[0] - origin[0], point[1] - origin[1]))
    if len(ordered) == 1:
        features.extend([0] * 42)
    features.append(len(ordered))
    return features


def isl_motion_features(hands):
    features = isl_features(hands)
    if features is None:
        raise ValueError("No hand detected.")
    origin = min(hands, key=lambda hand: hand["lmList"][0][0])["lmList"][0]
    return features + [origin[0], origin[1]]
