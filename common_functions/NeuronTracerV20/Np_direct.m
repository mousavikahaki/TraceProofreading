%This program compares the two ways to calculate Np
%It uses the original (not conformed arbors)

function [apair,dpair,zzz]=Np_direct(Ra,Rd,aid,did,apid,dpid,s)

aleng=length(apid);
dleng=length(dpid);

% tag axonal and dendritic branches
dpid(dpid==-1)=did(dpid==-1);
dtag=did;
for i=1:dleng
    dkids=find(dpid==i);
    if length(dkids)==1
        dtag(dkids)=dtag(i);
    end
end

apid(apid==-1)=aid(apid==-1);
atag=aid;
for i=1:aleng
    akids=find(apid==i);
    if length(akids)==1
        atag(akids)=atag(i);
    end
end

% calculation of segment lengths
aaa=sum((Ra-Ra(apid,:)).^2,2).^0.5;
ddd=sum((Rd-Rd(dpid,:)).^2,2).^0.5;

% narrowing down the number of segments, which could give rize to a potential synapse after all shifts within pre and post pixels
nu=1;
dseg=[];
for i=1:dleng
    C1=sum((ones(aleng,1)*(Rd(i,:)+Rd(dpid(i),:))-Ra-Ra(apid,:)).^2,2).^0.5-2*s-ddd(i)-aaa;
    abranches=find(C1.*aaa<0);
    if ~isempty(abranches) && ddd(i)>0
        dseg(nu)=i;
        asegs{nu}=abranches;
        nu=nu+1;
    end
end

% calculating five different distanses and leaving only potentially connected segments
apair=[]; dpair=[];
for i=1:length(dseg)
    ta=[]; td=[];
    atemp=asegs{i};
    atemp=atemp(abs(Rd(dseg(i),2)-Ra(atemp,2))<s+ddd(dseg(i))+aaa(atemp));
    if ~isempty(atemp)
        atemp=atemp(abs(Rd(dseg(i),1)-Ra(atemp,1))<s+ddd(dseg(i))+aaa(atemp));
        xyz=length(atemp);
        if xyz>0
            abr=(sum((ones(xyz,1)*(Rd(dseg(i),:)+Rd(dpid(dseg(i)),:))-Ra(atemp,:)-Ra(apid(atemp),:)).^2,2).^0.5<2*s+ddd(dseg(i))+aaa(atemp));
            xyz=nnz(abr);
            if xyz>0
                asegs_lim=atemp(abr);
                Rapdp=Ra(apid(asegs_lim),:)-ones(xyz,1)*Rd(dpid(dseg(i)),:);
                Raap=Ra(asegs_lim,:)-Ra(apid(asegs_lim),:);
                Rddp=ones(xyz,1)*(Rd(dseg(i),:)-Rd(dpid(dseg(i)),:));
                Radp=Ra(asegs_lim,:)-ones(xyz,1)*Rd(dpid(dseg(i)),:);
                Rapd=Ra(apid(asegs_lim),:)-ones(xyz,1)*Rd(dseg(i),:);
                dddd=ddd(dseg(i));
                aaaa=aaa(asegs_lim)';
                
                ta(1,:)=zeros(1,xyz); td(1,:)=sum(Rapdp.*Rddp,2)./dddd^2;
                ta(2,:)=ones(1,xyz);  td(2,:)=sum(Radp.*Rddp,2)./dddd^2;
                td(3,:)=zeros(1,xyz); ta(3,:)=-sum((Rapdp.*Raap),2)./(aaaa.^2)';
                td(4,:)=ones(1,xyz);  ta(4,:)=-sum((Rapd.*Raap),2)./(aaaa.^2)';
                ta=(1+ta.*sign(ta)-(ta-1).*sign(ta-1))./2; td=(1+td.*sign(td)-(td-1).*sign(td-1))./2;
                dist=(ones(4,1)*sum(Raap.^2')).*ta.^2+(ones(4,1)*sum(Rddp.^2')).*td.^2-2.*(ones(4,1)*sum((Raap.*Rddp)')).*ta.*td+2.*(ones(4,1)*sum((Rapdp.*Raap)')).*ta-2.*(ones(4,1)*sum((Rapdp.*Rddp)')).*td+ones(4,1)*sum(Rapdp.^2');
                [distopt, ~]=min(dist);
                
                testa=(sum((Raap.*Rddp)').*sum((Rapdp.*Rddp)')-sum((Rapdp.*Raap)').*dddd^2)./(aaaa.^2.*dddd^2-(sum((Raap.*Rddp)')).^2);
                testd=(-sum((Raap.*Rddp)').*sum((Rapdp.*Raap)')+sum((Rapdp.*Rddp)').*aaaa.^2)./(aaaa.^2.*dddd^2-(sum((Raap.*Rddp)')).^2);
                change=find(testa>0 & testa<1 & testd>0 & testd<1);
                if isempty(change)==0
                    distopt(change)=sum(Raap(change,:).^2').*testa(change).^2+sum(Rddp(change,:).^2').*testd(change).^2-2.*sum((Raap(change,:).*Rddp(change,:))').*testa(change).*testd(change)+2.*sum((Rapdp(change,:).*Raap(change,:))').*testa(change)-2.*sum((Rapdp(change,:).*Rddp(change,:))').*testd(change)+sum(Rapdp(change,:).^2');
                end
                
                an=find(distopt<=s^2);
                if isempty(an)==0
                    apair=[apair; asegs_lim(an)];
                    dpair=[dpair; dseg(i).*ones(length(asegs_lim(an)),1)];
                end
            end
        end
    end
end
zzz=sort(dtag(dpair).*100000+atag(apair));
