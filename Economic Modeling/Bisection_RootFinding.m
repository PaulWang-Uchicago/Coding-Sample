% Bisection Method Example
% This script finds the zero of a function f using the bisection method.

% Define the function for which to find the root (change as needed)
f = @(x) x.^3 - 4*x - 9;  % Example: f(x) = x^3 - 4x - 9

% Set the initial interval [a, b] such that f(a) and f(b) have opposite signs.
a = 2;  
b = 3;

% Check that the initial interval satisfies the bisection requirement
if f(a)*f(b) > 0
    error('f(a) and f(b) must have opposite signs.');
end

% Set tolerance and maximum iterations
tol = 1e-6;
max_iter = 100;

% Call the bisection method function
[root, iter] = bisection_method(f, a, b, tol, max_iter);

fprintf('The zero of the function is approximately: %f\n', root);
fprintf('Found in %d iterations.\n', iter);

% --- Function definition for bisection method ---
function [root, iterations] = bisection_method(f, a, b, tol, max_iter)
    iterations = 0;
    while iterations < max_iter
        iterations = iterations + 1;
        % Midpoint of the interval
        c = (a + b) / 2;
        % Check convergence: if the function value at midpoint is close enough
        % or the interval width is sufficiently small.
        if abs(f(c)) < tol || (b - a)/2 < tol
            root = c;
            return;
        end
        % Determine the subinterval in which the sign change occurs.
        if f(a) * f(c) < 0
            b = c;  % Root is in [a, c]
        else
            a = c;  % Root is in [c, b]
        end
    end
    % If max_iter reached without convergence, return the last midpoint.
    root = (a + b) / 2;
end
