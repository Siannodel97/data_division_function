{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a5164779-09f8-4eb0-b398-bf0dc448a94b",
   "metadata": {},
   "source": [
    "# Data Set Division"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f84dd765-4e51-477d-bab7-d7a4f1e81c8a",
   "metadata": {},
   "source": [
    "This Notebook will show you to divide our data set to fit our model avoiding overfitting and underfitting. Also, in the final there will be a defined function to do all the division at once &#128522;"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e36fa6a-be9c-4924-9969-ec10d7593759",
   "metadata": {},
   "source": [
    "# Set up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "90a95821-4867-496c-b925-6c820415724e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05f70fbd-b4dc-4da6-9273-6b27c06d90ff",
   "metadata": {},
   "source": [
    "# First division"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb4a07d5-7726-4ea1-b754-a0f2a98a8866",
   "metadata": {},
   "source": [
    "First of all, you need to separate the data set in 3 differents subsets: 1º for the fit, 2º for validation and 3º for test predictions. Sklearn has **train_test_split** function which do a random partition of our data set every time the script run. So, if we need 3 different subsets from our data set, we will run this function two time, one for the subset **train_split** and the subset **test_split**, and other one for the subset **val_set**.\n",
    "\n",
    "The **train_split**, like it's called, is for train our algorithm.\n",
    "\n",
    "The **val_set** is to check that our algorithm doesn't have **overfitting** or **underfitting**.\n",
    "\n",
    "The **test_set** is to check that our algorithm has a good accuracy for predict new samples."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db862f22-d887-428e-a386-5a72f9cbf347",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Imagine we have loaded our data set as \"df\", a DataFrame made by Pandas from our data file in .csv format.\n",
    "\n",
    "file_path = \"...\\folder\\data_set.csv\"\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# Now we separate our data set in 60% train set and 40% test set for example, it can be changed without problem\n",
    "# but it's better that train set > test set for a good prediction.\n",
    "train_set, train_set = train_test_split(df, test_size = 0.4, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0db4a742-9d79-4358-babf-3cc7750b22e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# We can see information about our new subsets\n",
    "train_set.info()\n",
    "train_set.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "325f700a-f33c-4138-83c9-2e7f8d253efc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now we will separate our test subset in validation set and test set\n",
    "val_set, test_set = train_test_split(test_set, test_size = 0.5, random_state= 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4b65913-7989-4590-b3a8-c3576e79c246",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Training set lenght:\",len(train_set))\n",
    "print(\"Validation set lenght:\",len(val_set))\n",
    "print(\"Test set lenght:\",len(test_set))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06f24147-96ec-4633-8f3a-915e6481e117",
   "metadata": {},
   "source": [
    "# Random partition and Stratified Sampling"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11daf06e-0bb1-423e-bbbf-0607068321ae",
   "metadata": {},
   "source": [
    "There is a parameter called **shuffle** in **train_test_split** function which can help to separate our data set if it is very big (Big Data reference &#128562;). **Shuffle** is used to avoid after many tries our algorithm \"doesn't watch\" all the data set by random mixing the data before the division."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e0037b0-a6e2-445a-8e42-edcfaee012c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# If shuffle = False, the function doesn't mix the data set before the partition\n",
    "# train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, shuffle=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ad40728b-a8f5-41f3-a46c-fc25c5879d08",
   "metadata": {},
   "source": [
    "If the data set isn't big enough, it can appear the risk to introduce **sampling bias**. **Sampling bias** is a type of bias that occurs when a sample selected for a study or analysis does not adequately represent the entire population. This leads to incorrect conclusions, as certain groups within the population may be overrepresented or underrepresented. \n",
    "\n",
    "To avoid that, sklearn introduces the **stratify** parameter which makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.\n",
    "\n",
    "For example, if variable \"y\" is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify = y will make sure that your random split has 25% of 0's and 75% of 1's."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b5e0cd9-87dd-4291-8f66-fe9b2b0b0dce",
   "metadata": {},
   "outputs": [],
   "source": [
    "# You need to change \"column\" for the feature which you want to hold the initial proportion\n",
    "# train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df[\"column\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7c0fcb18-58b2-4060-9338-7614f2a59041",
   "metadata": {},
   "source": [
    "# Defined Function for complete division"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef891663-ebc5-4e26-874e-c5a6ed4a1048",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_val_test_split(df, rstate = 42, shuffle = True, stratify = None):\n",
    "    strat = df[stratify] if stratify else None\n",
    "    train_set, test_set = train_test_split(\n",
    "        df, test_size = 0.4, random_state = rstate, shuffle = shuffle, stratify = strat)\n",
    "    strat = test_set[stratify] if stratify else None\n",
    "    val_set, test_set = train_test_split(\n",
    "        test_set, test_size = 0.5, random_state = rstate, shuffle = shuffle, stratifu = strat)\n",
    "    return (train_set, val_set, test_set)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f573695e-b667-4f7b-9cee-92ecc189f0a2",
   "metadata": {},
   "source": [
    "To check all of this work great, we can see the lenght of our set and subsets. Also, we can check that **stratify** hold the initial proportion of the feature chosen."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c8cf4816-51c2-4dc9-a4f1-8f9b282d3cda",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Change \"column\" for the feature to hold the initial proportion.\n",
    "train_set, val_set, test_set = train_val_test_split(df, stratify='column')\n",
    "print(\"Initial Data Set lenght:\", len(df))\n",
    "print(\"Training Set lenght:\", len(train_set))\n",
    "print(\"Validation Set lenght:\", len(val_set))\n",
    "print(\"Test Set lenght:\", len(test_set))\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "plt.figure(figsize=(10,10)\n",
    "df[\"column\"].hist()\n",
    "train_set[\"column\"].hist()\n",
    "val_set[\"column\"].hist()\n",
    "test_set[\"column\"].hist()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
