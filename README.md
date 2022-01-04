# DelayDiscounting
! Work In Progress

A Python implementation of the Delay Discounting neuropsychological task.


What is delay discounting?

Delay discounting is a measure of the degree to which a person prefers immediate rewards
relative to larger rewards that require a waiting period before receiving it. At one extreme,
a person would rather take a very small reward rather than waiting for a much larger one,
and at the other, a person would wait a very long time to receive a reward that is only
marginally higher than a reward that could be obtained immediately. Most people fall somewhere
in the middle, being willing to wait a relatively short amount of time when the difference
between the immediate and delay reward is large, but preferring the immediate reward when the
wait is long and the difference minimal. The point at which a person has no preference for either
is called the "indifference point", and when multiple such points have been calculated for
different time frames, a model can be fitted to vizualize how that individual discounts. The
shape of the model has been a point of interdisciplinary contention, with economists
traditionally holding that humans discount at an exponential rate, with longer delays being
disproportionately unpreferable. Conversely, neuroscientists have long held that humans discount
at a hyperbolic rate, with the preference for immediate rewards "levelling out" at a certain
point due to the limitations of the brain in conceptualizing points far in the future, which
leads us to penalize increased delays less harshly as the interval increases. Limited research
has been done to categorize individual test takers as either "more hyperbolic" or "more exponential"
based on the the closeness of the fitted model to the actual data.


What this package offers:

This program takes a series of delay/immediate preferences for various times and delay periods,
either from a csv or from administering the test, and computes indifference points for each
delay period. It uses these points to find the rate of discounting for both a hyperbolic
(delay amount / (1 + (discounting rate) * (delay length)) curve and an exponential curve
((delay amount) * e^( -(discounting rate) * (delay length) )). It also computes the
Residual Sum of Squares for each, and a trapezoidal area-under-the-curve for both
curves and the actual data points. The AUCs are compared and the subject is classified
as either a hyperbolic or exponential discounter based on which AUC is closer to the actual.
All of this information is printed to the command line.


Use:

Currently, the only option for use is to edit the main() function in the file itself. This
file uses an object-oriented approach, and an example of formatting and options is given
in the main function by default. The only thing that must be included is a subject name,
everything else has a default. If the program is run like this, it will immediately begin
delivering test prompts. Optionally, one can give a pre-formatted csv file with test responses.
The format is very specific: the first column must be dollar amounts (integers or floating points,
no dollar sign) representing the immediate reward amounts. The first row must be integers
representing the delay amount (in days). In each intersection, a 0 represents a preference
for the immediate reward, and a 1 represents a preference for the delay reward (if the delay
and immediate reward happen to be the same at any point, any number can be used as this cell
will be ignored). If given a csv, the program will immediately compute the statistics. If available,
the user can also supply a set of pre-calculated indifference points. If desired, the delay lengths
and immediate reward amounts can also be changed. To run, type the following on the command line:

python DelayDiscounting.py


Future additions:

Support for command line arguments

Storing computed information in a separate file

A cleaner test administration interface


NOTE: this program should not be used for diagnostic or research purposes.
