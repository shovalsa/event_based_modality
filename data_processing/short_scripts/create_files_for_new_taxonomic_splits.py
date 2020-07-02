import pandas as pd
import os


ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(os.getcwd())))[0]
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
# GME = pd.read_csv(os.path.join(ROOT_DIR, "data", "annotated_gme.csv"), sep="\t", keep_default_na=False)

def create_tax_level_files(tax_level_name, replacements_dict, fold):
    for root, dirs, files in os.walk(os.path.join(ROOT_DIR, "data", "GME", "finest", fold)):
        for file in files:
            if "finest_space.bmes" in file:
                with open(os.path.join(root, file), "r") as fine_f:
                    fname = file.replace("finest", subset_name)
                    with open(os.path.join(ROOT_DIR, "data", "GME", tax_level_name, fold, fname), "w") as binary_f:
                        for line in fine_f.readlines():
                            if line:
                                l = line.split()
                                try:
                                    token, prefix, suffix = l[0], l[1].split("-")[0], l[1].split("-")[1]
                                    for (super_label, sublabels) in replacements_dict:
                                        if any(x in l[1] for x in sublabels):
                                            binary_f.write(f"{token} {prefix}-{super_label}\n")
                                except:
                                    binary_f.write(line)
                            else:
                                binary_f.write(line)

                                
def create_tax_level_for_test(tax_level_name, replacements_dict):
    test_path = os.path.join(ROOT_DIR, "data", "GME", "finest", "test_finest_space.bmes")
    new_test_path = os.path.join(ROOT_DIR, "data", "GME", tax_level_name, "test_finest_space.bmes")
    with open(test_path, "r") as fine_f:
        with open(new_test_path, "w") as binary_f:
            for line in fine_f.readlines():
                if line.strip():
                    l = line.split()
                    token = l[0]
                    label = l[1].split("-")
                    if len(label) > 1:
                        prefix = label[0]
                        suffix = label[1]
                        for (super_label, sublabels) in replacements_dict:
                            if suffix in sublabels:
                                binary_f.write(f"{token} {prefix}-{super_label}\n")
                                break
                    else:
                        binary_f.write(line)
                else:
                    binary_f.write(line)
                                
                                
                        
def convert_data_to_taxonomy_level(tax_level_name, replacements_dict):
    gme_basepath = os.path.join(ROOT_DIR, "data", "GME")
    if not os.path.isdir(os.path.join(gme_basepath, tax_level_name)):
        os.mkdir(os.path.join(gme_basepath, tax_level_name))
    for fold in range(5):
        if not os.path.isdir(os.path.join(gme_basepath, tax_level_name, fold)):
            os.mkdir(os.path.join(gme_basepath, tax_level_name, fold))
    create_tax_level_for_test(tax_level_name, replacements_dict)
    for fold in range(5):
        create_tax_level_files(tax_level_name, replacements_dict, fold)
        
        
if __name__ == "__main__":
    first_tier = [
        ("priority", ["teleological", "deontic", "priority", "buletic"]),
        ("plausibility", ["epistemic", "circumstantial", "ability"])
    ]

    third_tier = [
      ("priority", ["priority", "buletic_teleological"]),
      ("deontic", ["deontic"]),
      ("epistemic", ["epistemic_circumstantial", "epistemic"]),
      ("ability", ["ability", "ability_circumstantial"]),
      ("buletic", ["buletic"]),
      ("teleological", ["teleological"]),
      ("circumstantial", ["circumstantial"]),
        ]

    m_not_m = [("modal", ["teleological", "deontic", "priority", "buletic", 
                         "epistemic", "circumstantial", "ability",
                         "epistemic_circumstantial", "buletic_teleological", "ability_circumstantial"])]


    plausibility_others = [
        ("deontic", ["deontic", "priority"]),
        ("intetional", ["buletic_teleological", "teleological", "buletic"]),
        ("plausibility", ["epistemic_circumstantial", "ability_circumstantial", "circumstantial",
                         "ability", "epistemic"])
    ]


    all_unrolled_no_ambiguities = [
        ("deontic", ["deontic", "priority"]),
        ("intetional", ["buletic_teleological", "teleological", "buletic"]),    
        ("circumstantial", ["circumstantial"]),
        ("ability", ["ability_circumstantial", "ability"]),
        ("epistemic", ["epistemic_circumstantial", "epistemic"])
    ]

    
    # this is an example how to call the converter. to add more versions, create a list of tuples as above.
    convert_data_to_taxonomy_level("plausibility_others", plausibility_others)

    
    
    
    
    
    