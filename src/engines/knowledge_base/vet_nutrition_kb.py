# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Knowledge Base for the Veterinary Nutrition RAG Engine.

This file contains the raw text data from veterinary nutrition guidelines,
specifically from the World Small Animal Veterinary Association (WSAVA) and
the Association of American Feed Control Officials (AAFCO). This data is
used to build the in-memory vector index for the RAG engine.
"""

WSAVA_NUTRITION_GUIDELINES = [
    {
        "source": "WSAVA Global Nutrition Guidelines",
        "content": "Nutritional assessment is a fundamental component of every patient evaluation. It is often considered the 'fifth vital sign,' alongside temperature, pulse, respiration, and pain assessment. Integrating a nutritional assessment into the routine physical examination provides a foundation for making specific nutritional recommendations for both healthy pets and those with diseases."
    },
    {
        "source": "WSAVA Global Nutrition Guidelines",
        "content": "The key components of a nutritional assessment include evaluating the animal, the diet, and the feeding management. This includes the patient's history, physical examination findings (body weight, body condition score, muscle condition score), and evaluation of the current diet. The goal is to identify any risk factors that could lead to nutritional problems."
    },
    {
        "source": "WSAVA Global Nutrition Guidelines",
        "content": "Body Condition Score (BCS) is a subjective assessment of body fat. The two most common scoring systems are a 5-point scale and a 9-point scale. On a 9-point scale, a BCS of 4-5 is ideal. A BCS of 1-3 indicates the pet is underweight, while 6-7 is overweight and 8-9 is obese. Ribs, lumbar vertebrae, and pelvic bones are easily palpable on underweight pets, while fat deposits are visible and palpable on overweight pets."
    },
    {
        "source": "WSAVA Global Nutrition Guidelines",
        "content": "Muscle Condition Score (MCS) assesses muscle mass, which can be lost even in overweight or obese animals (sarcopenia). MCS is graded as normal, or mild, moderate, or severe muscle loss. It is assessed by visualizing and palpating the spine, scapulae, skull, and wings of the ilia."
    },
    {
        "source": "WSAVA Global Nutrition Guidelines",
        "content": "When selecting a pet food, it is important to ensure it is complete and balanced for the pet's life stage. A nutritional claim on the pet food label should state that the food is 'complete and balanced' and for which life stage (e.g., growth, maintenance, gestation/lactation). This claim should be substantiated by one of two methods: formulation to meet AAFCO nutrient profiles or feeding trials according to AAFCO procedures."
    },
    {
        "source": "WSAVA Global Nutrition Guidelines",
        "content": "The WSAVA recommends asking the manufacturer specific questions to assess their quality control and nutritional expertise. These questions include whether they employ a full-time qualified nutritionist (PhD in animal nutrition or board-certified by ACVN or ECVCN), who formulates their diets, and what quality control measures are performed on their ingredients and finished products."
    },
    {
        "source": "WSAVA Global Nutrition Guidelines",
        "content": "Calculating Resting Energy Requirement (RER) is the first step in determining a pet's daily energy needs. The most common formula is: RER (kcal/day) = 70 * (body weight in kg)^0.75. For cats, a simpler linear formula can also be used: RER (kcal/day) = 40 * body weight in kg."
    },
    {
        "source": "WSAVA Global Nutrition Guidelines",
        "content": "Daily Energy Requirement (DER) is calculated by multiplying the RER by a factor based on the pet's life stage, activity level, and health status. For example, a typical neutered adult dog's DER is 1.6 * RER, while a weight loss plan might use 1.0 * RER. An active adult cat's DER might be 1.4 * RER."
    }
]

AAFCO_PET_FOOD_LABEL_GUIDELINES = [
    {
        "source": "AAFCO Pet Food Labeling Guide",
        "content": "The product name itself is regulated. If it includes a specific ingredient in the name (e.g., 'Chicken for Dogs'), that ingredient must make up at least 95% of the total product weight, excluding water. If a qualifier like 'dinner,' 'platter,' or 'entree' is used (e.g., 'Chicken Dinner'), the named ingredient must be at least 25% of the product. The 'with' rule (e.g., 'Dog Food with Chicken') requires only 3% of the named ingredient. The 'flavor' rule requires only that the flavor be detectable."
    },
    {
        "source": "AAFCO Pet Food Labeling Guide",
        "content": "The Guaranteed Analysis (GA) on a pet food label lists the minimum percentages of crude protein and crude fat, and the maximum percentages of crude fiber and moisture. The term 'crude' refers to the specific method of testing, not the quality of the nutrient. Be aware that the GA is on an 'as-fed' basis, which includes water. To compare foods with different moisture levels (e.g., dry vs. canned), you must convert them to a 'dry matter' basis."
    },
    {
        "source": "AAFCO Pet Food Labeling Guide",
        "content": "To convert Guaranteed Analysis to a Dry Matter (DM) basis, first find the percentage of moisture. If a food is 75% moisture, it is 25% dry matter. Then, divide the percentage of the nutrient (e.g., crude protein) by the percentage of dry matter. For example, a canned food with 10% protein and 75% moisture has a dry matter protein of 10 / (100-75) = 10 / 25 = 40% protein on a DM basis."
    },
    {
        "source": "AAFCO Pet Food Labeling Guide",
        "content": "The ingredient list must present ingredients in descending order by their pre-cooking weight. This includes their water content. Ingredients like 'chicken' or 'beef' contain a high percentage of water, which means they may appear higher on the list than a more concentrated dry ingredient like 'chicken meal,' even if the meal provides more actual protein."
    },
    {
        "source": "AAFCO Pet Food Labeling Guide",
        "content": "The Nutritional Adequacy Statement, also called the AAFCO statement, is one of the most important parts of a pet food label. It verifies that the food is 'complete and balanced' for a particular life stage. It must be substantiated either by formulation to meet an AAFCO Nutrient Profile (e.g., Dog Food Nutrient Profiles or Cat Food Nutrient Profiles) or by passing an AAFCO feeding trial."
    },
    {
        "source": "AAFCO Pet Food Labeling Guide",
        "content": "AAFCO does not approve, certify, or endorse pet foods. A statement like 'AAFCO Approved' is not permitted. The manufacturer is responsible for ensuring their product meets the AAFCO model regulations for nutritional adequacy. AAFCO provides the nutrient profiles and testing protocols, but does not perform any regulatory oversight of manufacturers."
    }
] 