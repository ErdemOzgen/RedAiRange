import hashlib
import json
from pathlib import Path
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot5
from packageurl import PackageURL
from cyclonedx.model import Property

# Function to generate SHA-256 hash of a file
def generate_sha256_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# Function to read dependencies from requirements.txt
def read_requirements(file_path):
    components = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split('>=') if '>=' in line else [line.strip(), '']
            name = parts[0].strip()
            version = parts[1].strip() if len(parts) > 1 else None
            components.append((name, version))
    return components

# Function to read model evaluations from model_evaluation.json
def read_model_evaluation(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to generate the ML SBOM
def generate_mlsbom(model_file_path, requirements_file_path, evaluation_file_path, output_file_path):
    bom = Bom()

    # Read and hash the model file
    model_hash = generate_sha256_hash(model_file_path)
    model_file_name = Path(model_file_path).name

    root_component = Component(
        name=model_file_name,
        type=ComponentType.MACHINE_LEARNING_MODEL,
        bom_ref='myModel',
        properties=[Property(name='hash', value=model_hash)]
    )
    
    # Add model evaluation properties to root component
    model_evaluations = read_model_evaluation(evaluation_file_path)
    for key, value in model_evaluations.items():
        root_component.properties.add(Property(name=key, value=str(value)))

    bom.metadata.component = root_component

    # Read components from requirements.txt and add them to the BOM
    components = read_requirements(requirements_file_path)
    for name, version in components:
        component = Component(
            name=name,
            version=version,
            type=ComponentType.LIBRARY,
            purl=PackageURL(type='pypi', namespace=None, name=name, version=version if version else None)
        )
        # Generate and add hash as a property
        if version:
            component_hash = generate_sha256_hash(requirements_file_path)  # Using file hash for this example
            component.properties.add(Property(name='sha256', value=component_hash))
        bom.components.add(component)
        bom.register_dependency(root_component, [component])

    # Serialize BOM to JSON using the CycloneDX library
    my_json_outputter = JsonV1Dot5(bom)
    serialized_json = my_json_outputter.output_as_string(indent=2)

    # Save the serialized JSON to a file
    with open(output_file_path, 'w') as f:
        f.write(serialized_json)

    print(f"SBOM generated and saved to {output_file_path}")

# How to use the function
model_file = 'simple-cifar10.h5'
requirements_file = 'requirements.txt'
evaluation_file = 'model_evaluation.json'
output_file = f'{model_file}.sbom.json'
generate_mlsbom(model_file, requirements_file, evaluation_file, output_file)
