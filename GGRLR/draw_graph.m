D=load('20news_w100'); 
S=cov(D.documents'); 
[W,invW,adj] = graphical_lasso(S, 0.005, 1/3, 100, eye(100));
adj = zeros(size(invW));
adj(abs(invW) > 2.5e-3) = 1;

[n,~]=size(adj);

R=5;
i=1:n;
x = R*cos(i/n*2*pi);
y = R*sin(i/n*2*pi);
xy = [x' y'];

h=clf;
gplot(adj, xy, 'k-'); hold on
for i=1:n
    h_text = text(0, 0, [D.wordlist{i}], 'color', 'b', 'fontsize', ...
         9, 'Rotation', 0);
    pos = [ get(h_text, 'Extent') get(h_text, 'Position')];
    width = 0;

    if i/n > 1/4 && i / n < 3/4
        rangle = i/n*360+180;
        width = pos(3)/0.7;
    else
        rangle = i/n * 360;
    end

    %disp(sprintf('%s %f', D.wordlist{i}, width));
    
    set(h_text, 'Rotation', rangle);
    set(h_text, 'Position', [(R + width)*cos(i/n*2*pi) (R+width)*sin(i/n*2*pi)]);
end
axis off equal
xlim([-6 6])
ylim([-6 6]);
%neg = zeros(size(adj)); neg(adj < 0) = 1;
%gplot(neg, xy, 'b-');

print(h, '-depsc', '-r300', 'word_relation.eps');