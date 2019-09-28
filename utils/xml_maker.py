import xml.etree.cElementTree as ET

age_list = [
    "xx-24",
    "25-34",
    "35-49",
    "50-xx"
    ]

gender_list = [
    "male",
    "female"
    ]

def make_xml(uid, age_group, gender, extrovert, neurotic, agreeable, conscientious, _open):
    root = ET.Element('root')

    ET.SubElement(root, 'user',
        id=str(uid),
        age_group=age_list[age_group],
        gender=gender_list[gender],
        extrovert=str(extrovert),
        neurotic=str(neurotic),
        agreeable=str(agreeable),
        conscientious=str(conscientious),
        open=str(_open))

    data = ET.tostring(root)
    
    with open(str(uid) + ".xml", "wb") as file_name:
        file_name.write(data)


if __name__ == "__main__":
    make_xml('123fsgfg45', 3, 0, 1.55, 2.3, 3.7, 1.1, 0.12)