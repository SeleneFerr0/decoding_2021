# -*- coding: utf-8 -*-

import re

def string_clean(l):
    
    a = re.findall('【e(\d+)(.*?)(\d+)】',l)
    if len(a)>0:
        b = re.search('【e(\d+)(.*?)(\d+)】',l).group(0)
        l = l.replace(b,'')
        
    a = re.findall('【(.*?)礼物】',l)
    if len(a)>0:
        b = re.search('【(.*?)礼物】',l).group(0)
        l = l.replace(b,'')
        
    a = re.findall('【买(.*?)】',l)
    if len(a)>0:
        b = re.search('【买(.*?)】',l).group(0)
        l = l.replace(b,'')
        
    a = re.findall('【618(.*?)】',l)
    if len(a)>0:
        b = re.search('【618(.*?)】',l).group(0)
        l = l.replace(b,'')
        
    a = re.findall('【七夕(.*?)】',l)  
    if len(a)>0:
        b = re.search('【七夕(.*?)】',l).group(0)
        l = l.replace(b,'')
        
        
    a = re.findall('【(.*?)女神(.*?)】',l)  
    if len(a)>0:
        b = re.search('【(.*?)女神(.*?)】',l).group(0)
        l = l.replace(b,'')        
        
        
    a = re.findall('【(.*?)专享价(.*?)】',l)  
    if len(a)>0:
        b = re.search('【(.*?)专享价(.*?)】',l).group(0)
        l = l.replace(b,'')           
        
    l = l.replace('￥','')
    l = l.replace('（三十年老店全网低价）','')
    l = l.replace('（三十年老店福利价）','')
    l = l.replace('#','')
    l = l.replace('!','')
    l = l.replace('@','')
    l = l.replace('※','')
    l = l.replace('§','')
    l = l.replace('【1份】','')
    l = l.replace('【2份】','')
    l = l.replace('【3份】','')
    l = l.replace('【5份】','')
    l = l.replace('【1小包】','')
    l = l.replace('【1+1】','')
    l = l.replace('（2-4天到）','')
    l = l.replace('【3份优惠装】','')
    l = l.replace('【520情人节】','')
    l = l.replace('【520礼物】','')
    l = l.replace('【全新升级】','')
    l = l.replace('【网红】','')
    l = l.replace('【首页领券】','')
    l = l.replace('【520花式告白送礼】','')
    l = l.replace('【59减10】','')
    l = l.replace('【七夕礼物】','')
    l = l.replace('【上海老牌国货】','')
    
    l = str(l).replace('【领券满99减20】', "")
    l = l.replace('（促）','')
    l = l.replace('（单点不送）','')
    l = l.replace('（含赠品）','')
    l = l.replace('(小红书同款桃桃)','')
    l = l.replace('小红书同款','')
    l = l.replace('（文字带非卖品，介意慎拍）','')
    l = l.replace('（新品）','')
    l = l.replace('（新）','')
    l = l.replace('（厂)','')
    l = l.replace('(次日达)','')
    l = l.replace('(款式随机)','')
    l = l.replace('(罐更优惠)','')
    l = l.replace('(随机味道)','')
    l = l.replace('➕','+')
    l = l.replace('（订金）','')
    l = l.replace('【领券满39减8】', "")
    l = l.replace('巧克刀', '巧克力')
    l = l.replace('【领券满89减8】', "")
    l = l.replace('【闪购福利】', "")
    l = l.replace('【泡面】', "")
    l = l.replace('【碗】', "")
    l = l.replace('【整箱】', "")
    l = l.replace('【整包】', "")
    l = l.replace('【大包】', "")
    l = l.replace('【单包】', "")
    l = l.replace('【五连包】', "")
    l = l.replace('【网红推荐】', "")
    l = l.replace('【网红爆款】', "")
    l = l.replace('【抖音爆款】', "")
    l = l.replace('【网红】', "")
    l = l.replace('【520情人节】','')
    l = l.replace('(送2粒七号电池)','')
    l = l.replace('（品牌随机）','')
    l = l.replace('（送砝码送2节7号电池）','')
    l = l.replace('【专柜直送】','')
    l = l.replace('【优选】','')
    l = l.replace('【入会专享价69.9元】','')
    l = strQ2B(l)
    l = l.replace("[EC]", "")
    l = l.replace("亚州", "亚洲")
    l = l.replace("$",'')
    l = l.replace("/",'')
    l = l.replace("@@",'')
    l = l.replace(" ",'')
    l = l.replace("（",'(')
    l = l.replace("）",')')
    l = l.replace(".",'')
    l = l.replace("^",'')
    l = l.replace("_",'')
    l = l.replace("?",'')
    l = l.replace("4合1",'四合一')
    l = l.replace("3合1",'三合一')
    l = l.replace("2合1",'二合一')
    l = l.replace("赠礼品袋",'')
    l = l.replace("3点1刻",'三点一刻')
    l = l.replace("LUZHOULAOJIAO",'')
    l = l.replace("LUZHOU",'')
    l = l.replace("[", "")
    l = l.replace("]", "")
    l = l.replace("', '", "/")
    l = l.replace("@", "")
    l = l.replace("##", "")
    l = l.replace("#", "")
    l = l.replace("****", "*")
    l = l.replace("***", "*")
    l = l.replace("**", "*")
    l = l.replace("spf+", "spf")
    l = l.replace("spf++", "spf")
    l = l.replace("spf+++", "spf")
    l = l.replace("亚州", "亚洲")
    l = l.replace("100年润发", "百年润发")
    l = l.replace("·","")
    l = l.replace('超值装套','超值装')
    l = l.replace('(专卖)','')
    l = l.replace('g盒','g')
    l = l.replace("^\w+$", "")
    l = l.replace('【首页领平台券】','')
    l = l.replace('【快客达优选】','')
    l = l.replace('【满99减45】','')
    l = l.replace('【爆款推荐】','')
    l = l.replace('【618必囤·买1享5！】','')
    l = l.replace('【2份】','')
    l = l.replace('【1份】','')
    l = l.replace('【送男友】','')
    l = l.replace('【中秋】','')
    l = l.replace('【年货】','')
    l = l.replace('【女神节礼物】','')
    l = l.replace('【ZHANG小柠檬】','')
    l = l.replace('【KKV】','')
    l = l.replace('【生日礼物】','')
    l = l.replace('特惠！','')
    l = l.replace('【520情人节】','')
    l = l.replace('【预售】','')
    l = l.replace('【新年礼物】','')
    l = l.replace('【端午节】','')
    l = l.replace('【因运输导致瓶身微微变形】','')
    l = l.replace('刮毛刀男女士专用脱毛去除腋毛腿毛私处阴毛全身修剪剃毛神器','')
    l = l.replace('冰镇清备注（啤酒饮料请提前备注或者告知）','')
    l = l.replace('【红书推荐】','')
    l = l.replace('(t)','')
    l = l.replace('(tt)','')
    l = l.replace('(jm)','')
    l = l.replace('(临)','')
    l = l.replace('(临促)','')
    l = l.replace('(临期促销)','')
    l = l.replace('(临期特价)','')
    l = l.replace('(二选一可备注)','')
    l = l.replace('收藏店铺随机赠送','')
    l = l.replace('(品牌随机)','')
    l = l.replace('(元旦礼物套装组合)','')
    
    # l = "".join(l.split())
    # move = dict.fromkeys((ord(c) for c in u"\xa0\n\t"))
    # l = l.translate(move)
    
    if l.startswith('*'):
        l = l[1:]
    if l.startswith('-'):
        l = l[1:]
        
    if l.startswith('a1'):
        l = l[2:]        
        
    l = l.replace('<>','')
    # l = re.sub('\d+', " ", l).strip()   #words with numbers
    # l = re.sub('[^A-Za-z]+', ' ',l)        #special characters
    l = re.sub(r'[0-9]+', '', l) #numbers
    
    return l

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                                       
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): 
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring



def string_clean2(l):
    l = str(l).replace('【领券满99减20】', "")
    if '此链勿拍' in l:
        l = ''
    l = l.replace('【KKV】','')
    l = l.replace('/份','')
    l = l.replace('超级推荐','')
    l = l.replace('⭐','')
    l = l.replace('【KK】','')
    l = l.replace('【一周期】','')
    l = l.replace('新年','')
    l = l.replace('情人节','')
    l = l.replace('礼物','')
    l = l.replace('男士女士通用','')
    l = l.replace('进口超市','')
    l = l.replace('强烈推荐','')
    l = l.replace('不挣钱交个朋友','')
    l = l.replace('【天猫旗舰品质】','')
    l = l.replace('不满意包退','')
    l = l.replace('96h持久留香','')
    l = l.replace('颜色分类:————下单即送精美浴球1个————','')
    l = l.replace('下单即赠同仁堂育发液☆☆勿拍☆☆','')
    l = l.replace('颜色分类:————下单即送精美浴球1个————','')
    l = l.replace('及时下手','')
    l = l.replace('家庭囤货装','')
    l = l.replace('抖音同款','')
    l = l.replace('承诺','')
    l = l.replace('戚薇同款','')
    l = l.replace('我全都要钜划算','')
    l = l.replace('成分安全','')
    l = l.replace('实际规格含量以sku名称显示为准','')
    l = l.replace('下单即送精美浴球1个','')
    l = l.replace('直播间','')
    l = l.replace('镇店','')
    l = l.replace('爆款','')
    l = l.replace('直播','')
    l = l.replace('包邮','')
    l = l.replace('限时秒杀','')
    l = l.replace('买2送1','')
    l = l.replace('【疯狂世界杯】','')
    l = l.replace('【夏日专享】','')
    l = l.replace('【宁波保税】','')
    l = l.replace('【正品承诺】','')
    l = l.replace('【秃头少女自救指南】','')
    l = l.replace('精选优货','')
    l = l.replace('临期','')
    l = l.replace('清仓','')
    l = l[l.find('元')+1:]
    l = l.replace('边牧','')
    l = l.replace('萨摩耶','')
    l = l.replace('第2件','')
    l = l.replace('半价','')
    l = l.replace('低至','')
    l = l.replace('包邮','')
    l = l.replace('送礼佳品','')
    l = l.replace('官方旗舰店','')
    l = l.replace('官网','')
    l = l.replace('【领券满39减8】', "")
    l = l.replace('巧克刀', '巧克力')
    l = l.replace('【领券满89减8】', "")
    l = l.replace('【闪购福利】', "")
    l = l.replace('【泡面】', "")
    l = l.replace('【碗】', "")
    l = l.replace('【整箱】', "")
    l = l.replace('【整包】', "")
    l = l.replace('【大包】', "")
    l = l.replace('【单包】', "")
    l = l.replace('【五连包】', "")
    l = l.replace('【网红推荐】', "")
    l = l.replace('【网红爆款】', "")
    l = l.replace('【抖音爆款】', "")
    l = l.replace('【网红】', "")
    l = l.replace('【双十一】', "")
    l = l.replace('【双11】', "")
    l = l.replace('囤货','')
    l = l.replace('大部分顾客选择','')
    l = l.replace('天猫','')
    l = l.replace('防伪码可查','')
    l = l.replace('1+1>2', '一加一>二')
    l = l.replace("【", "(")
    l = l.replace("】", ")")
    l = l.replace("（",'(')
    l = l.replace("）",')')
    l = l.replace("正品", "")
    l = l.replace('❤','')
    l = l.replace('✅','')
    l = l.replace('【买1送1】','')
    l = l.replace('【送护发素】','')
    l = l.replace('(送同款古龙香沐浴露)','')
    l = l.replace('【90天售后 不满意随时退】','')
    l = l.replace('【收藏送身体乳】','')
    l = l.replace('【收藏加购送六重礼】','')
    l =l.replace('【送替换泵头】','')
    l = l.replace('【咨询】','')
    l = l.replace('【送雪肤皂】','')
    l = l.replace('【送旅行装】','')
    l = l.replace('【收藏送头皮修护液】','')
    l = l.replace('收藏加购送【单品小样】','')
    l = l.replace('【收藏加购送礼品】','')
    l = l.replace('【送护发素】','')
    l = l.replace('【收藏加购】送洗衣留香珠','')
    l = l.replace('收藏加购送【5重赠品】全新升级香氛','')
    l = l.replace('收藏加购送【免洗护发胶囊3粒】买套装更划算','')
    l = l.replace('【收藏加购送礼品】送沐浴露30ml+炮弹发膜*3','')
    l = l.replace('加购送【洗护4件装】升级香氛','')
    l = l.replace('加购收藏送洗护4件套【TMALL甄选】高·高保障·高服务','')
    l = l.replace('收藏加购送按摩梳+15ml护发精油（旅行便携装）','')
    l = l.replace('——————收藏加购送精美浴花——————','')
    l = l.replace('【收藏加购】送按摩梳+发膜试用装','')
    l = l.replace('收藏加购送【炮弹发膜/顺丰优先发货】','')
    l = l.replace('【收藏加购】送气垫梳+护发精油试用装','')
    l = l.replace('收藏加购送【试用装10ml*4】买套装更划算','')
    l = l.replace('【二伏吃面】','')
    l = l.replace('新客福利【收❤️藏~~礼】送手帕纸/包（没收♥藏不送，多拍不送，只限1次哦）','')
    a = re.findall('【收藏送(.*?)】',l)
    if len(a)>0:
        a1 = re.search('【收藏送(.*?)】',l).group(0)
        l = l.replace(a1,'')
        
    a = re.findall('【收藏加购送(.*?)】',l)
    if len(a)>0:
        a1 = re.search('【收藏加购送(.*?)】',l).group(0)
        l = l.replace(a1,'')    

    a = re.findall('【加购收藏送(.*?)】',l)
    if len(a)>0:
        a1 = re.search('【加购收藏送(.*?)】',l).group(0)
        l = l.replace(a1,'')  
    
    
    a = re.findall('【送(.*?)】',l)
    if len(a)>0:
        a1 = re.search('【送(.*?)】',l).group(0)
        l = l.replace(a1,'')  
    
    a = re.findall('【赠(.*?)】',l)
    if len(a)>0:
        a1 = re.search('【赠(.*?)】',l).group(0)
        l = l.replace(a1,'')    
    if '(勿拍)' in l:
        l = ''
    if '(勿下单)' in l:
        l = ''
    # l = re.sub(r'\(([^()]+)\)', '', l)
    # l = l.replace("[EC]", "")
    l = l.replace("亚州", "亚洲")
    l = l.replace("$",'')
    l = l.replace("/",'')
    l = l.replace("@@",'')

    l = l.replace("【", "")
    l = l.replace("】", "")
    # l = l.replace(".",'')
    l = l.replace("^",'')
    l = l.replace('方便·携带','')
    l = l.replace("超值",'')
    l = l.replace("瓶装",'瓶')
    l = l.replace("掌柜",'')
    l = l.replace("男女士品牌授权",'')
    l = l.replace("品牌授权",'')
    l = l.replace("疑问咨询专业客服解答",'')
    l = l.replace("解答",'')
    l = l.replace("疑问",'')
    l = l.replace("专业客服解答",'')
    l = l.replace("咨询",'')
    # l = l.replace("_",'')
    # l = l.replace("?",'')
    l = l.replace("4合1",'四合一')
    l = l.replace("3合1",'三合一')
    l = l.replace("2合1",'二合一')
    l = l.replace("赠礼品袋",'')
    l = l.replace("3点1刻",'三点一刻')
    l = l.replace("LUZHOULAOJIAO",'')
    l = l.replace("LUZHOU",'')
    l = l.replace("[", "")
    l = l.replace("]", "")
    l = l.replace("', '", "")
    l = l.replace("@", "")
    l = l.replace("送亲友", "")
    l = l.replace("****", "*")
    l = l.replace("***", "*")
    l = l.replace("**", "*")
    l = l.replace("spf+", "spf")
    l = l.replace("spf++", "spf")
    l = l.replace("spf+++", "spf")
    l = l.replace("spf50+", "spf")
    l = l.replace("SPF+", "spf")
    l = l.replace("SPF++", "spf")
    l = l.replace("SPF+++", "spf")
    
    l = l.replace("限送1个沐浴球", "")
    l = l.replace("亚州", "亚洲")
    l = l.replace("100年润发", "百年润发")
    # l = l.replace("·","")
    l = l.replace('408307508','')
    l = l.replace('BG959880ASAB','')
    l = l.replace('拉布拉多','')
    l = l.replace('怀旧记忆','')
    l = l.replace('80后','')
    l = l.replace('多盒优惠','')
    l = l.replace('新旧包装','')
    l = l.replace('单盒','')
    l = l.replace('㊙️','')
    l = l.replace('!','')
    l = l.replace('，','')
    l = l.replace('！','')
    l = l.replace('现货！)','')
    l = l.replace('现货','')
    l = l.replace('好效期)','')
    l = l.replace('联系改','')
    l = l.replace('联系客服','')
    l = l.replace('改价','')
    l = l.replace('隐私','')
    l = l.replace('不限购','')
    l = l.replace('限购','')
    if l.startswith('2022'):
        l=l.replace('2022','')
    # l = l.replace('适用','')
    l = l.replace('随机发货','')
    l = l.replace('发货','')
    l = l.replace('男女通用','')
    l = l.replace('限量秒杀①:','')
    l = l.replace('限量','')
    l = l.replace('新款','')
    l = l.replace('⭐','')
    l = l.replace('旗舰店','')
    l = l.replace('夫妻同补','')
    l = l.replace('套餐⑥:','')
    l = l.replace('限时专享:','')
    l = l.replace('买1套送1套','')
    l = l.replace('【限量劲爆秒杀】','')
    
    l = l.replace("//",'')
    l = l.replace(';','')
    l = l.replace("全套礼盒❤:",'')
    l = l.replace("基础4件套:",'')
    l = l.replace("✅15送7件礼", '')
    l = l.replace("✅全家福", '')
    l = l.replace("超划算", '')
    l = l.replace("店长", '')
    l = l.replace("推荐", '')
    l = l.replace("大部分选", '')
    l = l.replace("大部分人选", '')
    l = l.replace("热卖款", '')
    l = l.replace("热卖", '')
    l = l.replace("军训", '')
    l = l.replace("心动组合", '')
    l = l.replace("秒杀", '')
    l = l.replace("全家福洁面", '洁面')
    l = l.replace("明星偏爱", '')
    l = l.replace("明星同款", '')
    l = l.replace("买5送1加", '')
    l = l.replace("买10送2加", '')
    l = l.replace("❤", '')
    l = l.replace('包邮','')
    l = l.replace("活动款B", '')
    l = l.replace("活动款A", '')
    #2022141110020126811481308280
    l = l.replace("研益生菌丨送","\+")
    # l = l.replace("^\w+$", "")
    l = l.replace("90%人选择", "")
    l = l.replace("\+共17件豪礼", "")
    l = l.replace("✅", '')
    l = l.replace("新品特惠到手仅119", '')
    l = l.replace("特惠", '')
    l = l.replace("新品", '')
    l = l.replace("领券立减", '')
    l = l.replace("+咨询立减", '')
    l = l.replace("领券更优惠", '')
    l = l.replace("咨询立减", '')
    l = l.replace("咨询大额领券", '')
    l = l.replace("以下品牌信息请勿拍", '')
    # l = l.replace("-", '')
    l = l.replace("请", '')
    l = l.replace("请优惠", '')
    l = l.replace('保障','')
    l = l.replace("人气之选", '')
    l = l.replace("保证(30天包退)", '')
    l = l.replace("保证", '')
    l = l.replace("假一赔十", '')
    l = l.replace('不参与拍送', '')
    l = l.replace("颜色分类:多规格可选净含量:0g", '')
    if l.endswith('\+豪礼'):
        l.replace('\+豪礼','')
    l = l.replace("产品已更换新包装新品牌名:雅得康", '')
    l = l.replace("✿礼盒款", '')
    l = l.replace("加购领优惠", '')
    l = l.replace("经典推荐", '')
    l = l.replace("经典搭档", '')
    l = l.replace("热销", '')
    l = l.replace('颜色分类:多款规格可选','')
    l = l.replace('(送20个一次性手套)颜色分类:洗护套装礼盒','')
    l = l.replace('蒲公英、绿茶、玫瑰果随机发一款','')
    l = l.replace('超高性价比','')
    l = l.replace('买就送','')
    l = l.replace('再送','')
    l = l.replace('拍套装更划算','')
    l = l.replace('收藏加购优先','')
    l = l.replace('优先','')
    l = l.replace('收藏','')
    l = l.replace('加购','')
    l = l.replace('重度脱发的选择','')
    l = l.replace('颜色分类:100%','')
    l = l.replace('主图款','')
    l = l.replace('主图同款','')
    
    '''TO LOWER'''
    l = l.lower()
    l = l.replace('pa+++','pa')
    l = l.replace('pa++','pa')
    l = l.replace('pa+','pa')
    
    l = l.replace('人气组合','')
    l = l.replace('大容量','')
    l = l.replace('实惠套装','')
    l = l.replace('官方','')
    l = l.replace('买1送','买1赠')
    l = l.replace('买一送','买一赠')
    l = l.replace('千克','kg')
    l = l.replace('KG','kg')
    l = l.replace('升千克','kg')
    l = l.replace('升kg','kg')
    l = l.replace('新包装','')
    l = l.replace('重庆连锁药房','')
    l = l.replace('★★','')
    l = l.replace('颜色分类:净含量和型号如图所示', '')
    l = l.replace('颜色分类:力士洗沐组合','')
    l = l.replace('收藏加购优先','')
    l = l.replace('买贵包赔','')
    l = l.replace('假一罚十','')
    l = l.replace('可自选','')
    l = l.replace('颜色分类:精选','')
    l = l.replace('女生','')
    a = re.findall('(\d+)瓶仅(\d+)瓶',l)
    if len(a)>0:
        a1 = re.search('(\d+)瓶仅(\d+)瓶',l).group(0)
        l = l.replace(a1,'')
    
    a =  re.findall('(\d+).(\d+)kg|升kg',l)
    if len(a)>0:
        a1 = re.search('(\d+).(\d+)kg|升kg',l).group(0)
        num = a[0][0]+ a[0][1]+'00'+'g'
        l = l.replace(a1,num)
        
    
    a =  re.findall('(\d+)%用户选择',l)
    if len(a)>0:
        a1 = re.search('(\d+)%用户选择',l).group(0)
        l = l.replace(a1,'')
    
    
    a =  re.findall('(\d+)%(.*?)选择',l)
    if len(a)>0:
        a1 = re.search('(\d+)%(.*?)选择',l).group(0)
        l = l.replace(a1,'')    
    
    
    a =  re.findall('(\d+)%(.*?)选',l)
    if len(a)>0:
        a = re.search('(\d+)%(.*?)选',l).group(0)
        l = l.replace(a,'')
        
    a =  re.findall('拍[一二三四五]发[一二三四五]',l)
    if len(a)>0:
        a = re.search('拍[一二三四五]发[一二三四五]',l).group(0)
        l = l.replace(a,'')
    
    
        
    l = l.replace('mL','ml')
    l = l.replace('毫升','ml')
    l = str(l).replace('♥♥', "")
    l = l.replace('加购收藏优先','')
    l = l.replace('+送浴球','')
    l = l.replace('香味:其他','')
    l = l.replace('颜色分类:多香型可选','')
    l = l.replace('+赠浴花','')
    l = l.replace('旗舰','')
    l = l.replace('店牌','')
    l = l.replace('店','')
    l = l.replace('授权','')
    l = l.replace('品牌','')
    l = l.replace(r'+送','+')
    l = l.replace('颜色分类:规格型号如图所示','')
    l = l.replace('颜色分类:便携旅行装出差旅行常备','')
    l = l.replace(r'+送','送')
    l = l.replace('颜色分类:多款式规格可选','')
    l = l.replace('颜色分类:多规格可选','')
    l = l.replace('颜色分类:直供保证','')
    l = l.replace('浴球颜色分类:','')
    l = l.replace('—','')
    l = l.replace('下单即送','')
    l = l.replace('赠品随机发','')
    l = l.replace('送运费险买贵包赔颜色分类:','')
    l = l.replace('送泵头','')
    l = l.replace('送运费险','')
    l = l.replace('送精美浴球','')
    l = l.replace('送浴球','')
    l = l.replace('#赠送90天运费险#','')
    l = l.replace('#','')
    l = l.replace('1次带走','')
    l = l.replace('40%用户选择','')
    l = l.replace('无赠品','')
    l = l.replace('实惠*','')
    l = l.replace('.颜色分类','')
    l = l.replace('☆','')
    l = l.replace('力荐','')
    l = l.replace('收藏铺优先','')
    l = l.replace('产品破损包赔补发','')
    l = l.replace('上新促销','')
    l = l.replace('颜色分类:天猫','')
    l = l.replace('颜色分类:金色','')
    l = l.replace('✨','')
    l = l.replace('促销','')
    l = l.replace('特惠','')
    l = l.replace('热荐','')
    l = l.replace('下单享','')
    l = l.replace('极速','')
    l = l.replace('客服','')
    l = l.replace('体验装','')
    l = l.replace('保密','')
    l = l.replace('全新升级柔顺黑科技','')
    l = l.replace('★','')
    l = l.replace('自营','')
    l = l.replace('勿拍不','')
    l = l.replace('享好礼','')
    l = l.replace('品质','')
    
    
    l = l.replace(' 套装更划算','')
    l = l.replace(' 任选(默认随机)','')
    l = l.replace('任选(','')
    l = l.replace('默认','')
    l = l.replace('新老版本随机发！','')
    l = l.replace(' 实发','')
    l = l.replace('享','')
    l = l.replace('限时优惠','')
    l = l.replace('》','')
    l = l.replace('《','')
    
    a = re.findall('购买规格:',l)
    if len(a)>0:
        a1 = re.search('购买规格:(\d+)件',l).group(0)
        b1 = re.search('购买规格:(\d+)',l).group(1)
        b1 = '*'+ b1
        l = l.replace(a1,b1)
    
    try:
        a = re.search('(\d+)ml[+](\d+)ml[+](\d+)ml洗发水(\d+)ml[+]护发素(\d+)ml[+]沐浴露(\d+)ml',l).group(0)
        b = re.search('(\d+)ml[+](\d+)ml[+](\d+)ml',l).group(0)
        l = l.replace(b,'')
    except AttributeError:
        pass
    
    l = l.replace('( 洗+护+沐+洁面四件套)密集滋养洗发水195g+密集滋养护发素195g+深层营润190g+粉润洁面乳30g','密集滋养洗发水195g+密集滋养护发素195g+深层营润190g+粉润洁面乳30g')
    if l[-3:] in ['送2样']:
        l = l[:-3]
    
    l = l.replace('()','')
    l = l.replace('#','')
    l = l.replace('全新升级','')
    l = l.replace('网红奶油卷点心蛋糕休闲零食小吃早餐','')
    return l
''

def label_sort(l):
    l = l.upper()
    if '\\' in l:
        l = l.replace('\\','/')
    ls = l.split('/')
    ls.sort()
    l = '/'.join(ls)
    return l