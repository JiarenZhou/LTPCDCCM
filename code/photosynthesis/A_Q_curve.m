function A =A_Q_curve(para,PPFD)

phi=para(1); %photosynthetic efficiency
P_max=para(2);% leaf photosynthetic rate under the saturated incident light intensity
Rd=para(3); % dark respiration rate
theta=para(4);% convexity of the A-Q curve.
A=(phi.*PPFD+P_max-sqrt((phi.*PPFD+P_max).^2-4*theta*phi.*PPFD*P_max))/(2*theta)-Rd;

end