import xml.etree.ElementTree as ET


def get_humanoid_geoms(mjcf_path):
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find(".//worldbody")

    def collect_geom_names(body_element):
        geom_names = []
        geom_names.extend(geom.get("name") for geom in body_element.findall("geom"))

        for child_body in body_element.findall("body"):
            geom_names.extend(collect_geom_names(child_body))

        return geom_names

    all_geom_names = []
    for body in worldbody.findall("body"):
        all_geom_names.extend(collect_geom_names(body))

    return all_geom_names


if __name__ == "__main__":
    print(get_humanoid_geoms("/home/spoonbobo/simulator/Assets/MJCF/humanoid.xml"))
