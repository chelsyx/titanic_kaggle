# Kaggle - Titanic

This is an article about what I did and what I learned from the kaggle project - [**Titanic: Machine Learning from Disaster**](https://www.kaggle.com/c/titanic). Although this is not my first kaggle project, and I thought I had a lot of experience in data analysis and modeling, I was surprised that there're so many great methods/tricks/packages I never used before, and so many bad habits I had but never realized, through doing this project and reading other people's solutions. Therefore, I feel I should at least summarize what I learned and reflect on my own solution.


## What I did

I categorize the predictors into 3 groups: 

- **Social-Economic Status**: Pclass, Fare; And 2 variables I created: Deck(CabinL) and wCabin(Cabin!="" --> wCabin=1). wCabin is problematic since wCabin=0 just represent missing values in "Cabin" according to [wehrley](https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md#background). 
- **Connections**: SibSp, Parch; And 3 variables I created: ticket\_dup(duplicated Ticket), ticket\_pal(number of people who share the same Ticket), relative(SibSp+Parch).
- **Other Demographics**: Sex, Age, Embarked

I always start doing feature engineering right away, together with data wrangling and data exploration, which cost a lot of time and it turns out that most of them were not useful. In the future, I should start with the original predictors and run a simple model to get an overview and baseline, then try to do some feature engineering according to domain knowledge or insight gained from the previous step if I have time.

I used `LiblineaR::LiblineaR` to fit a logistic regression model with L2 penalty. And I tried both `randomForest::randomForest` and `party::cforest` to fit a random forest. Using the same set of predictors, logistic regressions are always better than random forests in this project.

After removing insignificant/non-important variables from models, both logistics regression and random forest generate a better result! It looks like even though logit regression with a L2 penalty and random forest can handle feature selection automatically, sometimes it's worth to try selecting features manually.

The best result I got on leader board is 0.77990, using logistic regression with a L2 penalty.

## What I learned

### Data Exploration

This is a part that I oversee before. Good data explorations can be helpful to model more efficiently and reduce the chance of making mistakes!

Use aggregate tables(percentage is more useful than absolute values) and plots. First find the most influential predictor on the dependent variable, then combine it with the second influential one, then the third one, like doing decision tree manually, see how pure the classification is after each step. 

1, **Plot**

Choose appropriate plot according to the type of variables: discrete - bar chart; continuous - histogram; dis vs. con - box plot; dis vs. dis - bar plot/mosaic plots (`barplot(prop.table(table(x, y)))` or `vcd::mosaicplot`, reveal percentage); con vs. con - line graph/scatter.

2, **Aggregate Table**

- Use `prop.table(table(x,y))` to create proportion(row/column/all) tables
- `Hmisc::bystats` is an advanced version of `aggregate`
- Converting continuous variables to categorical variables(bins), then plot/aggregate proportions. It's a useful trick to combine various predictors, to check their interactions when doing data exploration(like manually building decision tree).

### Data Wrangling

1, **Recoding**

When read in data in the first step, define `na.string` and other data structures(like data type, etc.) in the read-in function can save a lot of work!

For factor, recoding number to character if you want to create a cross-table or confusion matrix, but the other way around if you want to create a line graph. See [here](https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md#data-munging). Two functions in R work for this purpose: `plyr::revalue` and `plyr::mapvalues`.

2, **Missing Values**

Use `Amilia::missmap` to display missing data's pattern, percentage, etc.

Tricks of imputation:

- When there're few NAs, just replace them with mean/median, or group mean(e.g. replace Fare NA by Pclass mean), or mode for categorical variables(e.g. Embarked)
- When there're a lot of NAs in categorical/character variables(e.g. Cabin), should code NA as "Unknown"
- When there're a lot of NAs in continuous variables(e.g. Age), use model to impute. I used `mice::mice` and `missForest::missForest`. [Trevor](http://trevorstephens.com/post/73770963794/titanic-getting-started-with-r-part-5-random) use decision tree(`rpart`, method="anova") to predict Age with other predictors.

It's worth noting that in original random forest algorithm, unlike CART(surrogate splits), we do have to impute missing values(median imputation and/or proximity based measure), although there seems to be other modifications on that. See [here](http://stats.stackexchange.com/questions/98953/why-doesnt-random-forest-handle-missing-values-in-predictors).

Imputation of training set and test set:

- Combining training set and test set, impute missing values in both set at the same time([Trevor](http://trevorstephens.com/post/73770963794/titanic-getting-started-with-r-part-5-random)). However, it's not useful in real world problems, because test set is future data.
- Use the same method to impute training and test set, but do it separately. In this way, when we evaluate model performance using test set, we're evaluating the imputation method also.
- Apply the imputation parameters from training set to test set, depending on the imputation method. For example, using mean in the training set to replace NAs in test set. 


3, **Outlier**

- After detecting suspected outlier, I should check other variables of these observations to further confirm. A good example is **Fare**. Fare=0 are outliers, because after checking their age, we can see they're not babies, and we should impute them. Fare>500 is not an outlier. From wehrley's solution, Fare is group fare for people who was not travelling alone, which means it should be divided by the size of the group (I should have realized this when checking pair-wise plots for all variables).


### Feature Engineering

1, Create feature according to previous model's result:
- From [Kaggle tutorial](https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii):
> we know Pclass had a large effect on survival, and it's possible Age will too. One artificial feature could incorporate whatever predictive power might be available from both Age and Pclass by multiplying them. This amplifies 3rd class (3 is a higher multiplier) at the same time it amplifies older ages. Both of these were less likely to survive, so in theory this could be useful.
- Trevor create a feature with FamilySize+LastName to further separate those who travel with a large group/closed people
- wehrley create finer categorical features like 3rd class&Mr. and female&<15(female and children policy), which may be more influential than the original ones
- When certain levels of a factor are insignificant in previous model, think about compress the insignificant ones(e.g. remove `Embarked` and add `I(Embarked="S")` to the model)

2, Class compression. Reasons: Tree based model cannot handle the variable when there are too few(<=2?) observations under certain labels(e.g. combining rare titles).

3, Combining training and test set when doing feature engineering. Reason: 1) use information from both set, like find out which titles could be combined together; 2) no need to update levels to factors in test set([Trevor](http://trevorstephens.com/post/73461351896/titanic-getting-started-with-r-part-4-feature))

4, Domain knowledge and background information. Deck(first letter of Cabin) and side of ship(last digit of Cabin is even or odd) could be useful if there is not that many unknowns in Cabin.

5, Linear combination of features(e.g. Sibsp+Parch) is unnecessary for linear regression, but useful for decision trees.

Many solutions extract titles from Name, which turns out to be a very influential feature.

And don't forget to plot the new feature to see if it meet your expectation and if they reveal more information!

### Fitting a Model

1, **Decision Tree**

- Be aware that decision tree favors factor with many levels. Think about class compression to avoid overfitting.
- Ways to tune a decision tree: 1)change complexity parameters to grow deeper or trim; 2)trim trees manually by selecting nodes that you want to kill([Trevor](http://trevorstephens.com/post/72923766261/titanic-getting-started-with-r-part-3-decision): `rattle::fancyRpartPlot`); 3)remove some predictors from the model

2, **Random Forest**

- Random forest cannot deal with factor with levels>32. Solution: recoding factors to reduce the levels(e.g. [Trevor](http://trevorstephens.com/post/73770963794/titanic-getting-started-with-r-part-5-random) increase the cut-off to be a "Small" family from 2 to 3 people)
- Forest of conditional inference tree(`party::cforest`)! It is not based on node purity, but statistical test. It can handle more factor levels.
- When sample size is small, complex model is not necessarily better. From almost everyone's solution, we can see that random forest doesn't perform better than logistic regression.

3, `caret::train` is super useful!!!

>This function sets up a grid of tuning parameters for a number of classification and regression routines, fits each model and calculates a resampling based performance measure.

In `caret::train` we can 1) choose the optimization metric(wehrley use ROC, which is actually AUC) 2) set up parameters of CV 3) training method(SVMs, AdaBoost) 4) preprocessing(center, scale) method within resampling.

There're also a lot of other useful functions in `caret`! `caret::confusionMatrix` provides a lot of related statistics. `caret:resamples` and `dotplot` can collect the optimization metrics from all CV results and create CIs for model comparison([wehrley](https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md#model-evaluation)).

4, **Logistic Regression**

We should normalize variables if using logistic regression with LASSO or ridge regression, because penalization will then treat different IVs on a more equal footing. See [here](http://stats.stackexchange.com/questions/86434/is-standardisation-before-lasso-really-necessary). As shown in [LibLineaR](https://cran.r-project.org/web/packages/LiblineaR/LiblineaR.pdf), should use the mean and std of training set to normalize test set.
```sh
s2=scale(xTest,attr(s,"scaled:center"),attr(s,"scaled:scale"))
```

5, **Cross Validation**

Cross validation should be repeated, as in [wehrley](https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md#fitting-a-model), because k-fold cv has fairly large variance(all data points were only predicted once). See [here](http://stats.stackexchange.com/questions/18348/differences-between-cross-validation-and-bootstrapping-to-estimate-the-predictio) and [here](http://stats.stackexchange.com/questions/82546/how-many-times-should-we-repeat-a-k-fold-cv).