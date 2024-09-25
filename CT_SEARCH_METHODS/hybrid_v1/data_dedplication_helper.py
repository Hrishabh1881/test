import string
string_punctuation = string.punctuation + " "


def create_attribute_mapping(query_df, deduplication_fields):
    final_mapping = {}
    for each_field in deduplication_fields:
        if each_field not in final_mapping:
            final_mapping[each_field] = {}
        df_values = [eval(val) for val in query_df[each_field]]
        if isinstance(df_values, list):
            for each_val in df_values:
                if isinstance(each_val, list):
                    for each_sub_val in each_val:
                        each_sub_val = each_sub_val.strip()
                        temp_field_val = each_sub_val.lower()#each_sub_val.translate(str.maketrans(string_punctuation ,' ' * len(string_punctuation))).lower()
                        if temp_field_val not in final_mapping[each_field]:
                            final_mapping[each_field][temp_field_val] = each_sub_val
                        else:
                            if len(each_sub_val) > len(final_mapping[each_field][temp_field_val]):
                                final_mapping[each_field][temp_field_val] = each_sub_val
                elif isinstance(each_val, str):
                    each_val = each_val.strip()
                    temp_field_val = each_val.lower()#each_val.translate(str.maketrans(string_punctuation ,' ' * len(string_punctuation))).lower()
                    if temp_field_val not in final_mapping[each_field]:
                        final_mapping[each_field][temp_field_val] = each_val
                    else:
                        if len(each_val) > len(final_mapping[each_field][temp_field_val]):
                            final_mapping[each_field][temp_field_val] = each_val
        elif isinstance(df_values, str):
            df_values = df_values.strip()
            temp_field_val = df_values.lower()#df_values.translate(str.maketrans(string_punctuation ,' ' * len(string_punctuation))).lower()
            if temp_field_val not in final_mapping[each_field]:
                final_mapping[each_field][temp_field_val] = df_values
            else:
                if len(df_values) > len(final_mapping[each_field][temp_field_val]):
                    final_mapping[each_field][temp_field_val] = df_values
    return final_mapping
