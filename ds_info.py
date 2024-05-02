import pandas as pd
import numpy as np

def entropy(attribute: pd.Series):
    sum = 0
    values_count = attribute.value_counts()
    total = attribute.shape[0]
    for i in attribute.unique():
        p = values_count[i] / total
        sum += p * np.log2(p)
    return sum * (-1)


def entropy_info(target: pd.Series, attribute: pd.Series):
    sum = 0
    target_total = target.shape[0]
    counts = attribute.value_counts()
    df = pd.concat([target, attribute], axis=1)

    for i in attribute.unique():
        value_count = counts[i]
        for j in target.unique():
            details = df.apply(lambda x: True if x[attribute.name] == i and x[target.name] == j else False, axis=1)
            num = len(details[details == True].index)
            if num != 0:
                sum -= (value_count / target_total) * (num / value_count) * np.log2(num / value_count)
    return sum


def convert_num_by_sturges(col_name: str, df: pd.DataFrame):
    col = df[col_name]
    bins = int(1 + np.log2(len(col.unique())))
    bin_width = col.max() / bins
    bins_labels = [i for i in range(bins + 1)]

    intervals = [0 + i * bin_width for i in range(bins + 1)]
    intervals.append(np.float64(np.inf))

    return pd.cut(df[col_name], intervals, labels=bins_labels)


def split_info(target: pd.Series, attribute: pd.Series):
    sum = 0
    total_target = target.shape[0]
    counts = attribute.value_counts()

    for value in attribute.unique():
        value_count = counts[value]
        if value_count != 0:
            sum -= (value_count / total_target) * np.log2(value_count / total_target)
    return sum


def gain(target: pd.Series, attribute: pd.Series):
    target_entropy = entropy(target)
    attribute_entropy = entropy_info(target, attribute)
    return target_entropy - attribute_entropy


def gain_ratio(attribute: pd.Series, target: pd.Series):
    return gain(target, attribute) / split_info(target, attribute)


def find_attr_max_gain_ratio(target_name, df: pd.DataFrame):
    main_attribute = ('', float('-inf'))
    for col in df:
        if col != target_name:
            attr_gain_ratio = gain_ratio(df[col], df[target_name])
            if attr_gain_ratio > main_attribute[1]:
                main_attribute = (col, attr_gain_ratio)

    return main_attribute[0]


def find_main_attribute_for_price(df: pd.DataFrame):
    df['Price'] = pd.cut(df['Price'], [0, 15000, 25000, np.inf], labels=['budget', 'medium', 'expensive'])

    df['Processor_Speed'] = pd.cut(df['Processor_Speed'],
                                   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                                   labels=['1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0'])

    df['Storage_Capacity'] = pd.cut(df['Storage_Capacity'], [0, 350, 600, 1000], labels=['Small', 'Medium', 'Large'])

    df['Screen_Size'] = pd.cut(df['Screen_Size'], [11, 14, 16, 17], labels=['Small', 'Medium', 'Large'])

    df['Weight'] = convert_num_by_sturges('Weight', df)

    main_attribute = find_attr_max_gain_ratio('Price', df)

    return main_attribute
